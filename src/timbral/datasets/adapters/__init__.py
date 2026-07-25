"""Dataset annotation adapter registry: dispatches to the corresponding adapter by dataset_name."""

from . import (audioset_strong, audioset_weak, birdvox_14sd, bsd, datased,
               db3v, dcase2024_task5, desed, esc50, fsd50k, fsdnoisy18k,
               hyenaset, realdesed, sonyc_ust, urbansound8k)
from .base import DatasetAnnotation

ADAPTERS = {
    "ESC-50": esc50.load_annotation,
    "FSD50K": fsd50k.load_annotation,
    "DataSED": datased.load_annotation,
    "UrbanSound8K": urbansound8k.load_annotation,
    "FSDnoisy18k": fsdnoisy18k.load_annotation,
    "BSD35K": bsd.load_annotation_bsd35k,
    "BSD10K": bsd.load_annotation_bsd10k,
    "SONYC-UST": sonyc_ust.load_annotation,
    "DESED": desed.load_annotation,
    "AudioSetStrong": audioset_strong.load_annotation,
    "AudioSetWeak": audioset_weak.load_annotation,
    "BirdVox-14SD": birdvox_14sd.load_annotation,
    "DB3V": db3v.load_annotation,
    "DCASE-2024-Task-5": dcase2024_task5.load_annotation,
    "HyenaSET": hyenaset.load_annotation,
    "RealDESED": realdesed.load_annotation,
}


def register_adapter(dataset_name: str, load_fn) -> None:
    """Register a new dataset adapter (for extending with new datasets or test injection)."""
    ADAPTERS[dataset_name] = load_fn


def load_annotation(dataset_name: str, dataset_dir: str) -> DatasetAnnotation:
    """Dispatch to the corresponding adapter by dataset_name; raises an error listing registered adapters if not found."""
    if dataset_name not in ADAPTERS:
        raise KeyError(f"No adapter registered for dataset {dataset_name}, "
                       f"registered adapters: {sorted(ADAPTERS)}")
    return ADAPTERS[dataset_name](dataset_dir)
