"""Repository root location: the single root-location mechanism for the
entire repo.

The repository root is marked by a ``.project-root`` file, from which each
component derives its own asset directory (currently the datasets
component's ``assets/datasets/splits/``).
"""

from pathlib import Path

from rootutils import find_root


def project_root() -> Path:
    """Return the repository root directory (searched upward from this
    file for the ``.project-root`` marker).
    """
    return find_root(search_from=__file__, indicator=".project-root")
