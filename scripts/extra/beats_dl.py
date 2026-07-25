"""Download the official BEATs OneDrive weights to a specified directory.

The official weights are only published via OneDrive public sharing links;
plain HTTP cannot download them anonymously. This script uses Playwright
Chromium to anonymously open the sharing page, capture the temporary signed
download URL from the ``/items/`` API response, and then stream the download
with requests (resumable downloads + SHA-256 verification + atomic write).

Independent of the ``timbral`` runtime: does not import timbral/torch/rootutils;
playwright is lazily imported inside the download path, so environments
without playwright can safely import this module.

Usage::

    python scripts/extra/beats_dl.py --dest /path/to/dir \\
        [--entries beats_iter1 ...] [--workers 3]
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
_PROGRESS_INTERVAL_SECONDS = 5
_MAX_DOWNLOAD_ATTEMPTS = 5
_MAX_RESOLVE_ATTEMPTS = 3
_RESOLVE_TIMEOUT_SECONDS = 60

_PRETRAINED_SIZE = 361_499_833
_FINETUNED_SIZE = 363_145_291


@dataclass(frozen=True)
class BeatsDownloadTarget:
    """The fixed download identity of one official BEATs weight file.

    Attributes:
        share_url: The OneDrive sharing link from the official README.
        official_name: The official file name returned by OneDrive; must match exactly after resolution.
        size: The exact byte count of the official file; must match exactly after resolution.
        sha256: The digest that must match after the download completes.
    """

    share_url: str
    official_name: str
    size: int
    sha256: str


BEATS_DOWNLOAD_TARGETS: dict[str, BeatsDownloadTarget] = {
    "beats_iter1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpmY7IHhgc9q0pT7Q?e=uQuisJ"
        ),
        official_name="BEATs_iter1.pt",
        size=_PRETRAINED_SIZE,
        sha256=(
            "b5f4cc10bcbff63a437c695f33389e64"
            "11513b3f7d5cdae8fb62b5005f4a1fcd"
        ),
    ),
    "fine_tuned_beats_iter1_cpt1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpuRfRZmco2XulmFw?e=f2INHa"
        ),
        official_name="BEATs_iter1_finetuned_on_AS2M_cpt1.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "e0e739e3670bfbb93c51adefb1d02981"
            "621397addc979d392aefd3dc53c22cab"
        ),
    ),
    "fine_tuned_beats_iter1_cpt2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpyMlTmnRh0Wp_Qgg?e=sgzv8H"
        ),
        official_name="BEATs_iter1_finetuned_on_AS2M_cpt2.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "2f3a7b65ab232c4f75570d4d17e21e5e"
            "bc34b3c40fe1a074f27d199e81354960"
        ),
    ),
    "beats_iter2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpwwEGgUyiI-jQyQw?e=1rP1RI"
        ),
        official_name="BEATs_iter2.pt",
        size=_PRETRAINED_SIZE,
        sha256=(
            "81a23e00aa4878d7e8627ded87ea697f"
            "b347c8ceffed21223e0398ed0fa34ad8"
        ),
    ),
    "fine_tuned_beats_iter2_cpt1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcp4l547zKa7xPqy8w?e=rsLdPr"
        ),
        official_name="BEATs_iter2_finetuned_on_AS2M_cpt1.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "3a120810c0f6dbfd50a7f48dc03ed077"
            "971a50cb2dbb7999695d5c700d03da45"
        ),
    ),
    "fine_tuned_beats_iter2_cpt2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcp5APbt_2bdIQvX0w?e=2cd2ry"
        ),
        official_name="BEATs_iter2_finetuned_on_AS2M_cpt2.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "08363b9b5eabeb47b0879c84145b27c6"
            "03e7e50c116a633fa5b98ade119fc354"
        ),
    ),
    "beats_iter3": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5"
        ),
        official_name="BEATs_iter3.pt",
        size=_PRETRAINED_SIZE,
        sha256=(
            "8d1b234032a9ccff353612dc6c209823"
            "46dc2968b205b79d97303eb5e77bfb34"
        ),
    ),
    "fine_tuned_beats_iter3_cpt1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcplb48ll1zIt82eWQ?e=XyxrX7"
        ),
        official_name="BEATs_iter3_finetuned_on_AS2M_cpt1.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "379369a41d0b3749f746cdcea8036de5"
            "06cb3aedecce84de7db0a75fda2a4fe7"
        ),
    ),
    "fine_tuned_beats_iter3_cpt2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcptb4S-CeJnlJGtZA?e=2FyDy3"
        ),
        official_name="BEATs_iter3_finetuned_on_AS2M_cpt2.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "08374f1cbd49143900b351bc81cd307d"
            "e386a11f8e609eb3862634e992068b55"
        ),
    ),
    "beats_iter3_plus_as20k": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpvdNz8-aYim60CIg?e=53V8pg"
        ),
        official_name="BEATs_iter3_plus_AS20K.pt",
        size=_PRETRAINED_SIZE,
        sha256=(
            "8008b126bb5e8ab08912c60c58847ed6"
            "76d32e64a5864c922356b7c2522fb2f8"
        ),
    ),
    "fine_tuned_beats_iter3_plus_as20k_cpt1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcp2YHUCT1uZx2Kysw?e=nvu1Dw"
        ),
        official_name="BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "2c366278dcf835e9bdefad4f7147b0ed"
            "ba4b940c59146fd05dc49a401fa82ff8"
        ),
    ),
    "fine_tuned_beats_iter3_plus_as20k_cpt2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcp092af0h7P3kXKFA?e=kUkPhN"
        ),
        official_name="BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt2.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "6d28b32bfa7bcaaf84ab834186581c2a"
            "360c6669e372e808d054cf0ef4d5c2d2"
        ),
    ),
    "beats_iter3_plus_as2m": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
        ),
        official_name="BEATs_iter3_plus_AS2M.pt",
        size=_PRETRAINED_SIZE,
        sha256=(
            "d43cbfad4d7b56381c061d7a24774f90"
            "8d4d94c72961f6eb1d9090ff18cd8d34"
        ),
    ),
    "fine_tuned_beats_iter3_plus_as2m_cpt1": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC"
        ),
        official_name="BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "7f9362028ac6e5c049e8dc314d87e90e"
            "4f82a15a8e472deb56af55d7f9b34d6a"
        ),
    ),
    "fine_tuned_beats_iter3_plus_as2m_cpt2": BeatsDownloadTarget(
        share_url=(
            "https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea"
        ),
        official_name="BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        size=_FINETUNED_SIZE,
        sha256=(
            "e5815275a04b6885e7b8af63d120b29b"
            "ffae2cd2225cf4915e1ec6d819d3022c"
        ),
    ),
}


def _sha256(path: Path) -> str:
    """Compute the file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(byte_count: int) -> str:
    """Format a byte count into a human-readable binary unit."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def _wait_for_metadata(
    page: Any,
    metadata: dict[str, Any],
    timeout_seconds: int,
) -> None:
    """Wait for the sharing page to return file metadata containing the signed URL."""
    deadline = time.monotonic() + timeout_seconds
    while not metadata and time.monotonic() < deadline:
        page.wait_for_timeout(100)


def _resolve_download_url(
    browser: Any,
    index: int,
    total: int,
    entry: str,
    target: BeatsDownloadTarget,
) -> str:
    """Anonymously open the sharing page and resolve the temporary signed download URL.

    The resolved file name and byte count must match the fixed identity
    exactly, to guard against upstream content changes or a mismatched link.

    Args:
        browser: The Playwright Chromium browser.
        index: The current entry's index.
        total: The total number of entries.
        entry: The entry name.
        target: The fixed download identity.

    Returns:
        The temporary signed download URL.

    Raises:
        RuntimeError: Resolution still fails after multiple attempts, or the resolved result doesn't match the fixed identity.
    """
    for attempt in range(1, _MAX_RESOLVE_ATTEMPTS + 1):
        page = browser.new_page()
        metadata: dict[str, Any] = {}

        def capture_metadata(response: Any) -> None:
            if "/items/" not in response.url or response.status != 200:
                return
            try:
                item = response.json()
            except Exception:
                return
            if (
                isinstance(item, dict)
                and item.get("name")
                and item.get("size") is not None
                and item.get("@content.downloadUrl")
            ):
                metadata.update(item)

        page.on("response", capture_metadata)
        try:
            page.goto(
                target.share_url,
                wait_until="domcontentloaded",
                timeout=_RESOLVE_TIMEOUT_SECONDS * 1000,
            )
            _wait_for_metadata(
                page,
                metadata,
                timeout_seconds=_RESOLVE_TIMEOUT_SECONDS,
            )
            resolved = dict(metadata)
        except Exception as error:
            if attempt == _MAX_RESOLVE_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to resolve public link: {entry} ({target.share_url})"
                ) from error
            resolved = {}
        finally:
            page.close()

        if resolved:
            # Identity verification lives outside the retry try block: a
            # name/size mismatch is a deterministic failure caused by an
            # upstream content change, so retrying is pointless — raise
            # immediately.
            resolved_name = Path(str(resolved["name"])).name
            resolved_size = int(resolved["size"])
            if (
                resolved_name != target.official_name
                or resolved_size != target.size
            ):
                raise RuntimeError(
                    f"{entry} resolved result does not match the fixed identity: "
                    f"{resolved_name} ({resolved_size} bytes), expected "
                    f"{target.official_name} ({target.size} bytes)."
                )
            print(
                f"[resolve {index:02d}/{total:02d}] {entry} <- "
                f"{resolved_name} ({_format_bytes(resolved_size)})",
                flush=True,
            )
            return str(resolved["@content.downloadUrl"])

        print(
            f"[resolve {index:02d}/{total:02d}] {entry} attempt {attempt} "
            "did not obtain metadata, retrying",
            flush=True,
        )

    raise RuntimeError(f"Failed to resolve public link: {entry} ({target.share_url})")


def _download_from_offset(
    session: requests.Session,
    download_url: str,
    partial_path: Path,
    expected_size: int,
    entry: str,
) -> None:
    """Resume streaming the download from the end of the temporary file."""
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(
        download_url,
        headers=headers,
        stream=True,
        timeout=(30, 120),
    ) as response:
        if response.status_code == 416 and offset == expected_size:
            return
        response.raise_for_status()

        if offset and response.status_code == 206:
            mode = "ab"
        else:
            offset = 0
            mode = "wb"

        downloaded = offset
        last_report_time = time.monotonic()
        with partial_path.open(mode) as output_file:
            for chunk in response.iter_content(
                chunk_size=_DOWNLOAD_CHUNK_SIZE
            ):
                if not chunk:
                    continue
                output_file.write(chunk)
                downloaded += len(chunk)
                current_time = time.monotonic()
                if (
                    current_time - last_report_time
                    >= _PROGRESS_INTERVAL_SECONDS
                ):
                    print(
                        f"[download] {entry} "
                        f"{downloaded / expected_size:6.1%} "
                        f"({_format_bytes(downloaded)}/"
                        f"{_format_bytes(expected_size)})",
                        flush=True,
                    )
                    last_report_time = current_time


def download_target(
    entry: str,
    target: BeatsDownloadTarget,
    download_url: str,
    dest: Path,
) -> Path:
    """Download a single entry to ``dest/<entry>.pt``, with resume support.

    Args:
        entry: The entry name.
        target: The fixed download identity.
        download_url: The already-resolved temporary signed download URL.
        dest: The destination directory.

    Returns:
        The final file path, verified by byte count and SHA-256.

    Raises:
        RuntimeError: An existing file's digest doesn't match, the temporary file is oversized, or download retries are exhausted.
    """
    final_path = dest / f"{entry}.pt"
    partial_path = dest / f"{entry}.pt.part"

    if final_path.exists():
        actual_sha256 = _sha256(final_path)
        if actual_sha256 == target.sha256:
            print(f"[skip] {entry} already complete", flush=True)
            return final_path
        raise RuntimeError(
            f"Existing file's SHA-256 does not match, refusing to overwrite: {final_path} "
            f"(actual {actual_sha256}, expected {target.sha256})"
        )

    if (
        partial_path.exists()
        and partial_path.stat().st_size > target.size
    ):
        raise RuntimeError(
            f"Temporary file exceeds expected size, refusing to continue: {partial_path}"
        )

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_from_offset(
                session=session,
                download_url=download_url,
                partial_path=partial_path,
                expected_size=target.size,
                entry=entry,
            )
            actual_size = partial_path.stat().st_size
            if actual_size != target.size:
                raise RuntimeError(
                    f"{entry} has incorrect size: current {actual_size}, "
                    f"expected {target.size}"
                )
            actual_sha256 = _sha256(partial_path)
            if actual_sha256 != target.sha256:
                partial_path.unlink()
                print(
                    f"[verify] {entry} local temporary file SHA-256 mismatch (actual "
                    f"{actual_sha256}, expected {target.sha256}), deleted, "
                    f"retrying download (attempt {attempt}/{_MAX_DOWNLOAD_ATTEMPTS})",
                    flush=True,
                )
                continue
            partial_path.replace(final_path)
            print(
                f"[done] {entry} ({_format_bytes(target.size)})",
                flush=True,
            )
            return final_path
        except Exception as error:
            if attempt == _MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"{entry} download failed, retried "
                    f"{_MAX_DOWNLOAD_ATTEMPTS} times"
                ) from error
            print(
                f"[retry] {entry}: {error} "
                f"(attempt {attempt}/{_MAX_DOWNLOAD_ATTEMPTS})",
                flush=True,
            )
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"{entry} download failed")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the official BEATs OneDrive weights to a specified directory"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for the download (created automatically)",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        choices=sorted(BEATS_DOWNLOAD_TARGETS),
        default=sorted(BEATS_DOWNLOAD_TARGETS),
        metavar="ENTRY",
        help="Entry names to download; defaults to all 15",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel downloads (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve signed URLs, download in parallel, and summarize the results."""
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    from playwright.sync_api import sync_playwright

    args.dest.mkdir(parents=True, exist_ok=True)
    entries = list(dict.fromkeys(args.entries))
    print(
        f"{len(entries)} entries total, destination directory: {args.dest}",
        flush=True,
    )

    futures: dict[Future[Path], str] = {}
    completed_paths: list[Path] = []
    failed_entries: list[tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                for index, entry in enumerate(entries, start=1):
                    target = BEATS_DOWNLOAD_TARGETS[entry]
                    download_url = _resolve_download_url(
                        browser=browser,
                        index=index,
                        total=len(entries),
                        entry=entry,
                        target=target,
                    )
                    future = executor.submit(
                        download_target,
                        entry,
                        target,
                        download_url,
                        args.dest,
                    )
                    futures[future] = entry
            finally:
                browser.close()

        for future in as_completed(futures):
            entry = futures[future]
            try:
                completed_paths.append(future.result())
            except Exception as error:
                failed_entries.append((entry, error))

    if failed_entries:
        details = "\n".join(
            f"- {entry}: {error}" for entry, error in failed_entries
        )
        raise RuntimeError(
            f"{len(failed_entries)} entries failed to download:\n{details}"
        )

    total_size = sum(path.stat().st_size for path in completed_paths)
    print(
        f"All done: {len(completed_paths)} files, "
        f"total size {_format_bytes(total_size)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
