"""Generate the default split file for a specified dataset via the registry.

Args:
    - dataset_name: str, the dataset name; must already be registered in timbral.datasets.split_generators.
    - dataset_dir: str, the dataset's source-file root directory; must be passed explicitly.
    - output: str, the output JSON path; defaults to
      `{repo root}/assets/datasets/splits/{dataset_name}/default.json`.

The actual logic lives under src/timbral/datasets/split_generators; this script only handles argument parsing and dispatch.
"""

import argparse
import sys

import rootutils

# Root setup: locate the repo root (.project-root) and inject the src import path, without loading the project .env
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=False,
                            dotenv=False, cwd=False)
sys.path.insert(0, str(ROOT / "src"))

from timbral.datasets import split_generators


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of arguments to parse; reads ``sys.argv`` when ``None``.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate the default split JSON for a dataset (dispatched via the registry)")
    parser.add_argument("--dataset_name", required=True,
                        help="Dataset name, e.g. ESC-50")
    parser.add_argument("--dataset_dir", required=True, help="Dataset source-file root directory")
    parser.add_argument("--output", default=None,
                        help="Output JSON path; defaults to the default split path under assets")
    return parser.parse_args(args)


def main() -> None:
    """Thin dispatch: arguments → registry dispatch → print summary."""
    args = parse_args()
    result = split_generators.generate(args.dataset_name, args.dataset_dir,
                                       output_path=args.output)
    print("OUTPUT:", result["output_path"])
    print("COUNTS:", result["counts"])
    print("VERIFY:", result["verify"])


if __name__ == "__main__":
    main()
