"""Tests for the gold set schema and loader.

The gold set is the yardstick, so a malformed entry does not fail loudly at eval
time, it quietly moves the number. Every rule here exists to stop a question
that would score wrongly rather than error:

  - an adversarial question with sources would be counted as a retrieval hit
  - a normal question with no sources can never hit, so it drags hit rate down
    while looking like a retriever problem
  - a duplicate id silently overwrites a result
  - an accepted source no longer in the corpus reads as a regression
"""

import pytest
import yaml

from claricyte.rag.gold import (
    GOLD_PATH,
    GoldQuestion,
    coverage,
    load_gold,
    unknown_sources,
)
from claricyte.vocab import CLASSES


def make_question(**overrides) -> GoldQuestion:
    base = dict(
        id="q1",
        question="What causes a reactive lymphocytosis?",
        label="Lymphocyte",
        sources=("PMC12734239",),
    )
    base.update(overrides)
    return GoldQuestion(**base)


def write_gold(tmp_path, entries):
    path = tmp_path / "gold.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


# --- schema ---------------------------------------------------------------


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="unknown label"):
        make_question(label="Promyelocyte")


def test_adversarial_question_may_not_name_sources():
    with pytest.raises(ValueError, match="adversarial"):
        make_question(adversarial=True, sources=("PMC1",))


def test_normal_question_must_name_a_source():
    with pytest.raises(ValueError, match="no accepted sources"):
        make_question(sources=())


def test_adversarial_question_needs_no_source():
    assert make_question(adversarial=True, sources=()).adversarial


def test_question_is_frozen():
    with pytest.raises(Exception):
        make_question().question = "reassigned"


# --- loading --------------------------------------------------------------


def test_load_reads_every_field(tmp_path):
    path = write_gold(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "Why so many smudge cells?",
                "label": "Lymphocyte",
                "sources": ["PMC8255663", "PMC8418501"],
                "expect": "Fragile lymphocytes, classically CLL.",
                "tags": ["morphology"],
            }
        ],
    )
    (question,) = load_gold(path)
    assert question.sources == ("PMC8255663", "PMC8418501")
    assert question.tags == ("morphology",)
    assert not question.adversarial


def test_duplicate_ids_are_rejected(tmp_path):
    entry = {
        "id": "q1",
        "question": "What causes a monocytosis?",
        "label": "Monocyte",
        "sources": ["PMC4203415"],
    }
    path = write_gold(tmp_path, [entry, dict(entry)])
    with pytest.raises(ValueError, match="duplicate question id"):
        load_gold(path)


def test_empty_file_loads_as_no_questions(tmp_path):
    path = tmp_path / "gold.yaml"
    path.write_text("", encoding="utf-8")
    assert load_gold(path) == []


# --- corpus agreement -----------------------------------------------------


def test_unknown_sources_reports_by_question():
    questions = [
        make_question(id="q1", sources=("PMC1", "PMC2")),
        make_question(id="q2", sources=("PMC2",)),
    ]
    assert unknown_sources(questions, {"PMC2"}) == {"q1": ("PMC1",)}


def test_unknown_sources_is_empty_when_all_present():
    assert unknown_sources([make_question()], {"PMC12734239"}) == {}


def test_coverage_counts_by_class_and_adversarial():
    questions = [
        make_question(id="q1", label="Lymphocyte"),
        make_question(id="q2", label="Lymphocyte"),
        make_question(id="q3", label="Basophil", adversarial=True, sources=()),
    ]
    counts = coverage(questions)
    assert counts["Lymphocyte"] == 2
    assert counts["Basophil"] == 1
    assert counts["adversarial"] == 1
    assert set(counts) == set(CLASSES) | {"adversarial"}


# --- the committed file ---------------------------------------------------


def test_the_shipped_gold_set_loads():
    """Catches a malformed entry at commit time rather than at eval time."""
    questions = load_gold(GOLD_PATH)
    assert questions
    assert any(q.adversarial for q in questions)


def test_shipped_sources_are_all_in_the_corpus():
    """A rebuild can drop an article, which turns a question into one nothing
    can hit. That reads as a retrieval regression, so catch it here."""
    from claricyte.rag.corpus import read_jsonl

    corpus = {chunk.source_id for chunk in read_jsonl("rag_data/corpus.jsonl")}
    assert unknown_sources(load_gold(GOLD_PATH), corpus) == {}
