"""Shared pytest config: rootutils locates the project root and injects the project root and src import paths."""

import sys

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=False,
                            dotenv=False, cwd=False)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
