"""Tests for the explanation engine.

explain() is pure and torch-free precisely so it can be tested like this: it
encodes the clinical logic (what must always be said, what may be said, when to
hedge, when to flag a contradiction) with no model in the loop.
"""

import pytest

from claricyte.explain import (
    MAX_DISCRETIONARY,
    _article,
    _describe_granules,
    _join,
    explain,
)


def result_from(**overrides):
    """A full prediction dict, confident everywhere, with overrides applied.

    Values are (value, confidence) pairs keyed by attribute, matching what
    predict.predict returns.
    """
    base = {
        "cell_size": ("big", 0.99),
        "cell_shape": ("round", 0.99),
        "nucleus_shape": ("segmented-multilobed", 0.99),
        "nuclear_cytoplasmic_ratio": ("low", 0.99),
        "chromatin_density": ("densely", 0.99),
        "cytoplasm_vacuole": ("no", 0.99),
        "cytoplasm_texture": ("clear", 0.99),
        "cytoplasm_colour": ("light blue", 0.99),
        "granule_type": ("coarse", 0.99),
        "granule_colour": ("purple", 0.99),
        "granularity": ("yes", 0.99),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- helpers


def test_article_picks_an_before_vowel():
    assert _article("eosinophil") == "an"
    assert _article("basophil") == "a"


def test_join_uses_oxford_comma():
    assert _join(["a"]) == "a"
    assert _join(["a", "b"]) == "a and b"
    assert _join(["a", "b", "c"]) == "a, b, and c"


# ------------------------------------------------------- granule folding


def test_granules_absent_reads_as_no_granules():
    fragment, spoke = _describe_granules(result_from(granularity=("no", 0.99)))
    assert fragment == "no granules"
    # Only granularity speaks; type and colour are irrelevant when there are none.
    assert spoke == ["granularity"]


def test_granules_present_folds_type_and_colour_into_one_clause():
    fragment, spoke = _describe_granules(result_from())
    assert fragment == "coarse purple granules"
    assert spoke == ["granularity", "granule_type", "granule_colour"]


def test_nil_granule_fields_are_skipped():
    """A nil type contributes no words, so it must not appear in `spoke`
    either, or it could drag the clause into a hedge it did not earn."""
    fragment, spoke = _describe_granules(
        result_from(granule_type=("nil", 0.20), granule_colour=("red", 0.99))
    )
    assert fragment == "red granules"
    assert spoke == ["granularity", "granule_colour"]


def test_describe_granules_returns_none_without_granularity():
    result = result_from()
    del result["granularity"]
    assert _describe_granules(result) is None


# ------------------------------------------------------- required (Set A)


def test_required_feature_is_stated():
    text = explain(result_from(), {}, "Basophil")
    assert "coarse purple granules" in text
    assert "Consistent with a basophil" in text


def test_violated_required_feature_is_flagged_as_atypical():
    """A basophil whose granules read red is a contradiction, not a feature to
    quietly omit. The point of the given-label mode is to show disagreement."""
    text = explain(result_from(granule_colour=("red", 0.99)), {}, "Basophil")
    assert "atypical for a basophil" in text
    assert "granule colour was read as 'red'" in text.lower()


def test_required_features_are_stated_even_with_no_contributions():
    """Set A does not depend on the contribution scores at all."""
    text = explain(result_from(), {}, "Eosinophil")
    assert "Consistent with an eosinophil" in text


# --------------------------------------------------- discretionary (Set B)


def test_only_positive_contributions_are_mentioned():
    text = explain(
        result_from(),
        {"cell_size": 1.0, "cytoplasm_vacuole": -5.0},
        "Lymphocyte",
    )
    assert "a large cell" in text
    assert "no vacuoles" not in text


def test_discretionary_features_are_capped():
    contributions = {
        "cell_size": 5.0,
        "cell_shape": 4.0,
        "nuclear_cytoplasmic_ratio": 3.0,
        "chromatin_density": 2.0,
        "cytoplasm_texture": 1.0,
    }
    text = explain(result_from(), contributions, "Lymphocyte")
    # Fragments are comma-joined, so count them via the leading clause.
    body = text.split(": ", 1)[1].rstrip(".")
    fragments = [f.strip() for f in body.replace(", and ", ", ").split(", ")]
    assert len(fragments) == MAX_DISCRETIONARY


def test_highest_contribution_wins_when_capped():
    contributions = {
        "cell_size": 0.1,
        "cell_shape": 9.0,
        "nuclear_cytoplasmic_ratio": 8.0,
        "chromatin_density": 7.0,
    }
    text = explain(result_from(), contributions, "Lymphocyte")
    assert "a large cell" not in text  # lowest score, cut by the cap
    assert "a round shape" in text


def test_no_features_falls_back_rather_than_emitting_an_empty_sentence():
    text = explain(result_from(), {}, "Lymphocyte")
    assert text == "No distinguishing features were identified for lymphocyte."


# ---------------------------------------------------------------- hedging


def test_low_confidence_feature_is_hedged():
    text = explain(
        result_from(cell_size=("big", 0.40)), {"cell_size": 1.0}, "Lymphocyte"
    )
    assert "a large cell" in text
    assert "Confidence was low for cell size." in text


def test_confident_feature_is_not_hedged():
    text = explain(
        result_from(cell_size=("big", 0.99)), {"cell_size": 1.0}, "Lymphocyte"
    )
    assert "Confidence was low" not in text


def test_granule_clause_hedges_on_its_weakest_speaker():
    """Granules used to skip the confidence check entirely, so a coin-flip
    granule colour was asserted as flatly as a 0.99 one."""
    text = explain(result_from(granule_colour=("purple", 0.51)), {}, "Basophil")
    assert "coarse purple granules" in text
    assert "Confidence was low for granule colour." in text


def test_granule_clause_ignores_confidence_of_fields_it_did_not_use():
    """A nil granule_type at 0.20 says nothing, so it must not trigger a hedge."""
    text = explain(
        result_from(granule_type=("nil", 0.20), granule_colour=("purple", 0.99)),
        {},
        "Basophil",
    )
    assert "purple granules" in text
    assert "Confidence was low" not in text


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_hedge_threshold_is_honoured(threshold):
    text = explain(
        result_from(cell_size=("big", 0.5)),
        {"cell_size": 1.0},
        "Lymphocyte",
        low_confidence=threshold,
    )
    assert ("Confidence was low" in text) is (threshold == 1.0)
