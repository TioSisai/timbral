"""DB3V annotation adapter: species directory name -> multiclass weak."""

import os

from .base import DatasetAnnotation

# Single source of truth for the audio directory layout, referenced directly by the split generator
AUDIO_DIR = "data_wav_8s_2"


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load DB3V directory-based annotations and construct the species label for each WAV file.

    DB3V has no standalone annotation file; audio follows the
    ``data_wav_8s_2/{region}/{species}/{filename}.wav`` layout, where the three
    regions share the same class space and the species directory name is the
    single label.

    Args:
        dataset_dir: DB3V dataset root directory.

    Returns:
        DatasetAnnotation: Weak annotations whose path keys exactly match the
        relative paths used in the split.
    """
    weak_labels = {}
    classes = set()
    audio_root = os.path.join(dataset_dir, AUDIO_DIR)

    with os.scandir(audio_root) as region_entries:
        for region_entry in region_entries:
            if not region_entry.is_dir():
                continue
            with os.scandir(region_entry.path) as species_entries:
                for species_entry in species_entries:
                    if not species_entry.is_dir():
                        continue
                    species = species_entry.name
                    with os.scandir(species_entry.path) as audio_entries:
                        for audio_entry in audio_entries:
                            if (not audio_entry.is_file()
                                    or not audio_entry.name.lower().endswith(".wav")):
                                continue
                            audio_path = (
                                f"{AUDIO_DIR}/{region_entry.name}/"
                                f"{species}/{audio_entry.name}"
                            )
                            weak_labels[audio_path] = species
                            classes.add(species)

    return DatasetAnnotation(
        label_kind="multiclass",
        annotation_kind="weak",
        classes=sorted(classes),
        weak_labels=weak_labels,
    )
