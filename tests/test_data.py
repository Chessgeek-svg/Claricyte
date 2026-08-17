"""Tests for dataset loading.

MorphologyDataset only opens images in __getitem__, so everything the
constructor does (the merge, path normalization, class filtering, attribute
validation) can be tested from CSVs alone, with no image files on disk.
"""

import pandas as pd
import pytest

from claricyte.data import MorphologyDataset
from claricyte.vocab import ATTRIBUTE_VOCAB, ATTRIBUTES


def write_csvs(tmp_path, rows):
    """Write an attributes/metadata CSV pair and return their paths.

    Each row is a dict of overrides; anything omitted gets a legal default.
    """
    attributes, metadata = [], []
    for i, overrides in enumerate(rows):
        path = overrides.get("image_path", f"data\\cells\\cell_{i}.jpg")
        attribute_row = {attr: values[0] for attr, values in ATTRIBUTE_VOCAB.items()}
        attribute_row.update(
            {k: v for k, v in overrides.items() if k in ATTRIBUTE_VOCAB}
        )
        attribute_row["image_path"] = path
        attribute_row["source"] = "wbcatt"
        attributes.append(attribute_row)
        metadata.append(
            {
                "image_path": path,
                "source_dataset": "acevedo",
                "original_label": "whatever",
                "claricyte_label": overrides.get("claricyte_label", "Lymphocyte"),
                "split": overrides.get("split", "train"),
            }
        )

    attributes_path = tmp_path / "attributes.csv"
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(attributes).to_csv(attributes_path, index=False)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    return str(attributes_path), str(metadata_path)


def test_loads_and_encodes_attributes(tmp_path):
    dataset = MorphologyDataset(*write_csvs(tmp_path, [{}, {}]))
    assert len(dataset) == 2
    # Values are encoded to integer head indices, not left as strings.
    for attr in ATTRIBUTES:
        assert dataset.df[attr].iloc[0] == 0


def test_windows_paths_are_normalized_to_forward_slashes(tmp_path):
    """ingest.py builds paths with pathlib on Windows, so they arrive with
    backslashes. On Linux a backslash is an ordinary filename character, so the
    hosted demo would resolve every path to nothing."""
    dataset = MorphologyDataset(*write_csvs(tmp_path, [{}]))
    assert "\\" not in dataset.df["image_path"].iloc[0]
    assert dataset.df["image_path"].iloc[0].startswith("data/cells/")


def test_split_filter_selects_only_that_split(tmp_path):
    paths = write_csvs(
        tmp_path, [{"split": "train"}, {"split": "val"}, {"split": "val"}]
    )
    assert len(MorphologyDataset(*paths, split="val")) == 2
    assert len(MorphologyDataset(*paths, split="train")) == 1
    assert len(MorphologyDataset(*paths)) == 3


def test_index_is_reset_after_filtering(tmp_path):
    """app.py mixes label and positional indexing against this frame, which is
    only safe because the index is contiguous from zero."""
    paths = write_csvs(tmp_path, [{"split": "train"}, {"split": "val"}])
    dataset = MorphologyDataset(*paths, split="val")
    assert list(dataset.df.index) == [0]


def test_out_of_taxonomy_classes_are_dropped_with_a_warning(tmp_path):
    """The first attribute-labeled Blast would otherwise raise a bare KeyError
    from CLASS_TO_INDEX partway through an epoch."""
    paths = write_csvs(
        tmp_path, [{"claricyte_label": "Lymphocyte"}, {"claricyte_label": "Blast"}]
    )
    with pytest.warns(UserWarning, match="not in vocab.CLASSES"):
        dataset = MorphologyDataset(*paths)
    assert len(dataset) == 1


def test_unknown_attribute_value_raises_at_load_time(tmp_path):
    """Not mid-epoch as an int64 cast error, which is what used to happen."""
    paths = write_csvs(tmp_path, [{}, {"nucleus_shape": "banana"}])
    with pytest.raises(ValueError, match="outside vocab.ATTRIBUTE_VOCAB"):
        MorphologyDataset(*paths)


def test_missing_attribute_value_raises_at_load_time(tmp_path):
    paths = write_csvs(tmp_path, [{}, {"granularity": None}])
    with pytest.raises(ValueError, match="outside vocab.ATTRIBUTE_VOCAB"):
        MorphologyDataset(*paths)


def test_validation_error_names_the_offending_column_and_image(tmp_path):
    paths = write_csvs(
        tmp_path, [{}, {"image_path": "data\\bad.jpg", "cell_size": "enormous"}]
    )
    with pytest.raises(ValueError) as excinfo:
        MorphologyDataset(*paths)
    message = str(excinfo.value)
    assert "cell_size" in message
    assert "enormous" in message
    assert "bad.jpg" in message


def test_dropped_classes_are_not_validated(tmp_path):
    """Rows filtered out for being off-taxonomy should not then raise for their
    attribute values, or adding a class would become a two-step fight."""
    paths = write_csvs(
        tmp_path,
        [{}, {"claricyte_label": "Blast", "nucleus_shape": "banana"}],
    )
    with pytest.warns(UserWarning):
        dataset = MorphologyDataset(*paths)
    assert len(dataset) == 1
