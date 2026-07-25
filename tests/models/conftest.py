"""models test configuration: locate the project root, load the environment, and register alignment entries."""

import sys

import pytest
import rootutils

ROOT = rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=False,
    dotenv=True,
    cwd=False,
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_ALIGNMENT_ENTRIES = ("panns", "ast", "clap", "beats")


def pytest_addoption(parser):
    """Register the extensible model-alignment option."""
    parser.addoption(
        "--run-alignment",
        nargs="+",
        choices=_ALIGNMENT_ENTRIES,
        default=[],
        metavar="ENTRY",
        help=(
            "Run the official alignment tests for the given model(s); "
            "multiple entries can be supplied at once."
        ),
    )


def pytest_configure(config):
    """Register the alignment marker."""
    config.addinivalue_line(
        "markers",
        "alignment(entry): official alignment test requiring an explicit "
        "--run-alignment ENTRY",
    )


def pytest_collection_modifyitems(config, items):
    """Skip official alignment tests that were not explicitly enabled."""
    enabled_entries = set(config.getoption("--run-alignment"))
    for item in items:
        marker = item.get_closest_marker("alignment")
        if marker is None:
            continue
        entry = marker.args[0]
        if entry not in enabled_entries:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Requires explicitly passing --run-alignment {entry}."
                    )
                )
            )
