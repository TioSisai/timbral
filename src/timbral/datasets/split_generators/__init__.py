"""Default split generator registry: dispatches to the generator module of each dataset_name.

Structurally mirrors the adapters registry: each supported dataset has one generator
module, and each module is only responsible for building the split dict; writing to
disk and read-back verification are handled uniformly by this module's ``generate``.
"""

from pathlib import Path

from timbral.paths import project_root

from . import (audioset_strong, audioset_weak, base, birdvox_14sd, bsd,
               datased, db3v, dcase2024_task5, desed, esc50, fsd50k,
               fsdnoisy18k, hyenaset, realdesed, sonyc_ust, urbansound8k)

GENERATORS = {
    "ESC-50": esc50.generate,
    "FSD50K": fsd50k.generate,
    "DataSED": datased.generate,
    "UrbanSound8K": urbansound8k.generate,
    "FSDnoisy18k": fsdnoisy18k.generate,
    "BSD35K": bsd.generate_bsd35k,
    "BSD10K": bsd.generate_bsd10k,
    "SONYC-UST": sonyc_ust.generate,
    "DESED": desed.generate,
    "AudioSetStrong": audioset_strong.generate,
    "AudioSetWeak": audioset_weak.generate,
    "BirdVox-14SD": birdvox_14sd.generate,
    "DB3V": db3v.generate,
    "DCASE-2024-Task-5": dcase2024_task5.generate,
    "HyenaSET": hyenaset.generate,
    "RealDESED": realdesed.generate,
}


def register_generator(dataset_name: str, generate_fn) -> None:
    """Register a new split generator (for extending to new datasets or test injection)."""
    GENERATORS[dataset_name] = generate_fn


def default_split_path(dataset_name: str) -> Path:
    """Default split file path corresponding to dataset_name."""
    return (project_root() / "assets" / "datasets" / "splits"
            / dataset_name / "default.json")


def generate(dataset_name: str, dataset_dir: str, output_path=None) -> dict:
    """Generate the default split for dataset_name, write it to disk, and verify by
    reading it back.

    Args:
        dataset_name: Dataset name, used for registry dispatch.
        dataset_dir: Root directory of the dataset source files.
        output_path: Output JSON path; when ``None``, writes to the default split path.

    Returns:
        dict: ``output_path`` (the path written to), ``counts`` (entry count per
        split), and ``verify`` (read-back verification summary).

    Raises:
        KeyError: dataset_name has no registered split generator.
    """
    if dataset_name not in GENERATORS:
        raise KeyError(f"No split generator registered for dataset {dataset_name}, "
                       f"registered: {sorted(GENERATORS)}")
    if output_path is None:
        output_path = default_split_path(dataset_name)
    splits = GENERATORS[dataset_name](dataset_dir)
    counts = base.write_split_json(output_path, splits)
    verify_info = base.verify_split_json(output_path)
    return {"output_path": str(output_path), "counts": counts,
            "verify": verify_info}
