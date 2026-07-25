"""Unit tests for the DB3V adapter's pure directory logic."""

from timbral.datasets.adapters import db3v


_SPECIES = [
    "Agelaius phoeniceus",
    "Cardinalis cardinalis",
    "Certhia americana",
    "Corvus brachyrhynchos",
    "Molothrus ater",
    "Setophaga aestiva",
    "Setophaga ruticilla",
    "Spinus tristis",
    "Tringa semipalmata",
    "Turdus migratorius",
]


def test_db3v_directory_labels_match_split_paths(tmp_path):
    """Verify that paths in all three regions map exactly to the species directory name and cover the full class set."""
    expected_labels = {}
    for index, species in enumerate(_SPECIES):
        region = str(index % 3 + 1)
        filename = f"recording_{index}.wav"
        audio_dir = tmp_path / "data_wav_8s_2" / region / species
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / filename).touch()
        expected_labels[
            f"data_wav_8s_2/{region}/{species}/{filename}"
        ] = species

    # A path for the same species in a different region must not lose the
    # region or change the label.
    duplicate_species_dir = (
        tmp_path / "data_wav_8s_2" / "3" / _SPECIES[0]
    )
    duplicate_species_dir.mkdir(parents=True, exist_ok=True)
    (duplicate_species_dir / "cross_region.WAV").touch()
    expected_labels[
        f"data_wav_8s_2/3/{_SPECIES[0]}/cross_region.WAV"
    ] = _SPECIES[0]

    # A non-audio file is not a split item and should not enter the label mapping.
    (duplicate_species_dir / "notes.txt").touch()

    annotation = db3v.load_annotation(str(tmp_path))

    assert annotation.weak_labels == expected_labels
    assert annotation.classes == sorted(_SPECIES)
    assert annotation.label_kind == "multiclass"
    assert annotation.annotation_kind == "weak"
    assert annotation.strong_events is None
