"""Tests for the attribute vocabulary.

vocab is the single source of truth for the label space: the data pipeline
encodes training targets through it and inference decodes predictions through
it. If the two ever disagree, every prediction silently means the wrong thing,
so the round-trip properties below are the ones that actually matter.
"""

import numpy as np
import pandas as pd
import pytest

from claricyte import vocab


def test_attributes_matches_vocab_keys():
    assert vocab.ATTRIBUTES == list(vocab.ATTRIBUTE_VOCAB)
    assert len(vocab.ATTRIBUTES) == 11


@pytest.mark.parametrize("attribute", vocab.ATTRIBUTE_VOCAB)
def test_encode_decode_round_trip(attribute):
    """Every value survives value -> index -> value unchanged."""
    for value in vocab.ATTRIBUTE_VOCAB[attribute]:
        assert vocab.decode(attribute, vocab.encode(attribute, value)) == value


@pytest.mark.parametrize("attribute", vocab.ATTRIBUTE_VOCAB)
def test_indices_are_dense_and_ordered(attribute):
    """Indices are 0..n-1 in list order, which is what the head's output units
    assume. A gap or a reorder would remap every label ever encoded."""
    values = vocab.ATTRIBUTE_VOCAB[attribute]
    assert [vocab.encode(attribute, v) for v in values] == list(range(len(values)))
    assert vocab.num_classes(attribute) == len(values)


def test_no_duplicate_values_within_an_attribute():
    """A duplicate would make VALUE_TO_INDEX silently drop the earlier index."""
    for attribute, values in vocab.ATTRIBUTE_VOCAB.items():
        assert len(values) == len(set(values)), attribute


def test_encode_rejects_unknown_value():
    with pytest.raises(KeyError):
        vocab.encode("cell_size", "enormous")


def test_class_index_round_trip():
    for index, name in enumerate(vocab.CLASSES):
        assert vocab.CLASS_TO_INDEX[name] == index
    assert len(set(vocab.CLASSES)) == len(vocab.CLASSES)


def _clean_row():
    """One row holding the first legal value of every attribute."""
    return {attr: values[0] for attr, values in vocab.ATTRIBUTE_VOCAB.items()}


def test_find_invalid_rows_accepts_clean_data():
    df = pd.DataFrame([_clean_row(), _clean_row()])
    assert vocab.find_invalid_rows(df).empty


def test_find_invalid_rows_flags_out_of_vocab_value():
    rows = [_clean_row(), _clean_row()]
    rows[1]["nucleus_shape"] = "banana"
    invalid = vocab.find_invalid_rows(pd.DataFrame(rows))
    assert list(invalid.index) == [1]


def test_find_invalid_rows_flags_missing_value():
    """NaN.isin(...) is False, so missing values must come back as invalid.
    This is the case that used to surface as a cryptic int64 cast error deep
    inside __getitem__ instead of at load time."""
    rows = [_clean_row(), _clean_row()]
    rows[1]["granularity"] = np.nan
    invalid = vocab.find_invalid_rows(pd.DataFrame(rows))
    assert list(invalid.index) == [1]


def test_find_invalid_rows_ignores_absent_columns():
    """A frame missing an attribute column entirely is not the concern here;
    the column check is skipped rather than raising."""
    df = pd.DataFrame([{"cell_size": "small"}])
    assert vocab.find_invalid_rows(df).empty
