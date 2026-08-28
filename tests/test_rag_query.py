"""Tests for query construction.

Pure and torch-free, so the retrieval logic is testable with no model, no index
and no network. The properties here are the ones a silent failure would cost:
the GENERAL escape hatch (without it every smear-review passage is unreachable),
and the rule that only abnormal findings enter the query.
"""

import pytest

from claricyte.rag.corpus import GENERAL
from claricyte.rag.query import (
    CONDITION_TERMS,
    LOW_CONFIDENCE,
    build_query,
    class_filter,
    notable_findings,
)
from claricyte.vocab import CLASSES


def result_from(**overrides):
    """A full prediction dict, confident everywhere, with overrides applied."""
    base = {
        "cell_size": ("big", 0.99),
        "cell_shape": ("round", 0.99),
        "nucleus_shape": ("segmented-multilobed", 0.99),
        "nuclear_cytoplasmic_ratio": ("low", 0.99),
        "chromatin_density": ("densely", 0.99),
        "cytoplasm_vacuole": ("no", 0.99),
        "cytoplasm_texture": ("clear", 0.99),
        "cytoplasm_colour": ("light blue", 0.99),
        "granule_type": ("small", 0.99),
        "granule_colour": ("pink", 0.99),
        "granularity": ("yes", 0.99),
    }
    base.update(overrides)
    return base


# --- the filter -----------------------------------------------------------


def test_filter_matches_the_class_or_general():
    where = class_filter("Basophil")
    assert where == {
        "$or": [
            {"cell_classes": {"$contains": "Basophil"}},
            {"cell_classes": {"$contains": GENERAL}},
        ]
    }


@pytest.mark.parametrize("label", CLASSES)
def test_general_is_always_reachable(label):
    """Without this, pre-analytical and smear-review chunks tag to no class and
    can never be retrieved through any filter."""
    clauses = class_filter(label)["$or"]
    assert {"cell_classes": {"$contains": GENERAL}} in clauses


# --- notable findings -----------------------------------------------------


def test_vacuoles_are_reported_when_present():
    assert notable_findings(result_from(cytoplasm_vacuole=("yes", 0.9))) == [
        "cytoplasmic vacuolation"
    ]


def test_absent_vacuoles_are_not_reported():
    """Absence is uninformative: nobody writes a paper about cells lacking
    vacuoles, so the phrase would only add noise to the embedding."""
    assert notable_findings(result_from(cytoplasm_vacuole=("no", 0.99))) == []


def test_low_confidence_findings_are_ignored():
    below = LOW_CONFIDENCE - 0.1
    assert notable_findings(result_from(cytoplasm_vacuole=("yes", below))) == []


def test_normal_defining_features_never_enter_a_query():
    """Every neutrophil has pink granules, so saying so narrows nothing. Only
    attributes in NOTABLE_FINDINGS may appear."""
    text, _ = build_query(result_from(), "Segmented Neutrophil")
    for word in ("pink", "granule", "chromatin", "nucleus", "cytoplasm"):
        assert word not in text.lower()


# --- query text -----------------------------------------------------------


@pytest.mark.parametrize("label", CLASSES)
def test_class_name_always_appears_in_the_query(label):
    """The class is the highest-signal token for a corpus organised by cell type."""
    text, _ = build_query(result_from(), label)
    assert label.lower() in text.lower()


@pytest.mark.parametrize("label", CLASSES)
def test_condition_terms_are_included(label):
    text, _ = build_query(result_from(), label)
    for term in CONDITION_TERMS[label]:
        assert term in text.lower()


def test_query_is_prose_not_an_attribute_list():
    """The asymmetry fix: the query has to read like the documents it searches."""
    text, _ = build_query(result_from(), "Eosinophil")
    assert text.endswith(".")
    assert text.count(",") <= 2


def test_user_question_leads_but_keeps_the_class():
    text, _ = build_query(result_from(), "Monocyte", question="why is this not a blast")
    assert "why is this not a blast" in text
    assert "monocyte" in text.lower()


def test_findings_are_appended_to_a_user_question():
    text, _ = build_query(
        result_from(cytoplasm_vacuole=("yes", 0.99)),
        "Monocyte",
        question="what does this suggest",
    )
    assert "what does this suggest" in text
    assert "vacuolation" in text


def test_every_class_has_condition_terms():
    assert set(CONDITION_TERMS) == set(CLASSES)
