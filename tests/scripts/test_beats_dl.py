"""Network-free unit tests for ``scripts/extra/beats_dl.py``.

Loads the script module by file path via importlib (avoiding triggering the
playwright import), exercises the download function against a local HTTP
server with Range support, and asserts that the script's identity table is
consistent with ``timbral.models.helpers.beats``.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import sys
import threading

import pytest

from timbral.models.helpers.beats import BEATS_CHECKPOINTS
from timbral.paths import project_root

_SCRIPT_PATH = project_root() / "scripts" / "extra" / "beats_dl.py"


@pytest.fixture(scope="module")
def beats_dl():
    spec = importlib.util.spec_from_file_location(
        "beats_dl",
        _SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator requires the module to already be registered
    # in sys.modules
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Minimal test handler supporting Range resumable downloads.

    When ``payload_sequence`` is non-empty, each request consumes one
    response body from it (simulating recovery after bad data); once
    exhausted, it falls back to the fixed ``payload``.
    """

    payload: bytes = b""
    payload_sequence: list[bytes] = []

    def do_GET(self):  # noqa: N802 (http.server interface naming)
        cls = type(self)
        data = (
            cls.payload_sequence.pop(0)
            if cls.payload_sequence
            else cls.payload
        )
        range_header = self.headers.get("Range")
        if range_header is None:
            self.send_response(200)
            body = data
        else:
            start = int(range_header.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def range_server():
    _RangeRequestHandler.payload_sequence = []
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _RangeRequestHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/file"
    server.shutdown()
    thread.join()


def _make_target(beats_dl, payload: bytes):
    return beats_dl.BeatsDownloadTarget(
        share_url="https://1drv.ms/u/s!unused",
        official_name="unused.pt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


@pytest.fixture()
def payload():
    return bytes(range(256)) * 4096


def test_identity_table_matches_helpers(beats_dl):
    targets = beats_dl.BEATS_DOWNLOAD_TARGETS
    assert set(targets) == set(BEATS_CHECKPOINTS)

    for entry, target in targets.items():
        metadata = BEATS_CHECKPOINTS[entry]
        assert target.sha256 == metadata.sha256
        assert target.share_url.startswith("https://1drv.ms/u/")
        assert target.official_name.endswith(".pt")
        assert (
            "finetuned" in target.official_name
        ) == metadata.finetuned
        expected_size = (
            beats_dl._FINETUNED_SIZE
            if metadata.finetuned
            else beats_dl._PRETRAINED_SIZE
        )
        assert target.size == expected_size


def test_fresh_download(beats_dl, range_server, payload, tmp_path):
    _RangeRequestHandler.payload = payload
    target = _make_target(beats_dl, payload)

    result = beats_dl.download_target(
        "beats_iter1",
        target,
        range_server,
        tmp_path,
    )

    assert result == tmp_path / "beats_iter1.pt"
    assert result.read_bytes() == payload
    assert not (tmp_path / "beats_iter1.pt.part").exists()


def test_resume_from_partial(beats_dl, range_server, payload, tmp_path):
    _RangeRequestHandler.payload = payload
    target = _make_target(beats_dl, payload)
    partial = tmp_path / "beats_iter1.pt.part"
    partial.write_bytes(payload[: len(payload) // 2])

    result = beats_dl.download_target(
        "beats_iter1",
        target,
        range_server,
        tmp_path,
    )

    assert result.read_bytes() == payload


def test_existing_complete_file_skipped(
    beats_dl,
    payload,
    tmp_path,
):
    target = _make_target(beats_dl, payload)
    final = tmp_path / "beats_iter1.pt"
    final.write_bytes(payload)

    result = beats_dl.download_target(
        "beats_iter1",
        target,
        "http://127.0.0.1:1/unreachable",
        tmp_path,
    )
    assert result == final


def test_existing_corrupt_file_rejected(beats_dl, payload, tmp_path):
    target = _make_target(beats_dl, payload)
    (tmp_path / "beats_iter1.pt").write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        beats_dl.download_target(
            "beats_iter1",
            target,
            "http://127.0.0.1:1/unreachable",
            tmp_path,
        )


def test_oversized_partial_rejected(beats_dl, payload, tmp_path):
    target = _make_target(beats_dl, payload)
    (tmp_path / "beats_iter1.pt.part").write_bytes(payload + b"extra")

    with pytest.raises(RuntimeError, match="refusing to continue"):
        beats_dl.download_target(
            "beats_iter1",
            target,
            "http://127.0.0.1:1/unreachable",
            tmp_path,
        )


def test_sha_mismatch_then_redownload_succeeds(
    beats_dl,
    range_server,
    payload,
    tmp_path,
):
    _RangeRequestHandler.payload = payload
    _RangeRequestHandler.payload_sequence = [b"x" + payload[1:]]
    target = _make_target(beats_dl, payload)

    result = beats_dl.download_target(
        "beats_iter1",
        target,
        range_server,
        tmp_path,
    )

    assert result.read_bytes() == payload
    assert not (tmp_path / "beats_iter1.pt.part").exists()


def test_sha_mismatch_rejected(
    beats_dl,
    range_server,
    payload,
    tmp_path,
    monkeypatch,
):
    _RangeRequestHandler.payload = payload
    target = beats_dl.BeatsDownloadTarget(
        share_url="https://1drv.ms/u/s!unused",
        official_name="unused.pt",
        size=len(payload),
        sha256="0" * 64,
    )
    monkeypatch.setattr(beats_dl, "_MAX_DOWNLOAD_ATTEMPTS", 1)

    with pytest.raises(RuntimeError, match="download failed"):
        beats_dl.download_target(
            "beats_iter1",
            target,
            range_server,
            tmp_path,
        )
    assert not (tmp_path / "beats_iter1.pt").exists()
    # The corrupt .part has been deleted; re-running can restart the
    # download from scratch
    assert not (tmp_path / "beats_iter1.pt.part").exists()


def test_size_mismatch_rejected(
    beats_dl,
    range_server,
    payload,
    tmp_path,
    monkeypatch,
):
    _RangeRequestHandler.payload = payload[:-100]
    target = _make_target(beats_dl, payload)
    monkeypatch.setattr(beats_dl, "_MAX_DOWNLOAD_ATTEMPTS", 1)

    with pytest.raises(RuntimeError, match="download failed"):
        beats_dl.download_target(
            "beats_iter1",
            target,
            range_server,
            tmp_path,
        )
