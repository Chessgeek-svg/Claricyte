"""Tests for the live question path.

pipeline is the one module that joins retrieval to generation, so both ends are
stubbed here: search is monkeypatched and the provider is injected. What is
actually under test is the joining, and the two decisions in it that differ from
the offline panel script:

  - a fabricated citation is reported, not raised, since a live answer that is
    mostly sound is better shown with a warning than swallowed
  - an empty retrieval abstains without ever calling the provider, which would
    otherwise be a paid call with nothing to ground it
"""

import pytest

from claricyte.rag import pipeline
from claricyte.rag.corpus import Chunk
from claricyte.rag.generate import ABSTAIN


def make_chunk(index: int) -> Chunk:
    return Chunk(
        text=f"Body text {index}.",
        source_id="PMC1",
        section="Discussion",
        url="https://example.org/PMC1",
        license="CC BY",
        cell_classes=("Monocyte",),
        title="Monocytosis in practice",
        chunk_index=index,
    )


class FakeProvider:
    """Returns a canned answer and records what it was asked."""

    def __init__(self, answer: str = "Monocytes rise in chronic infection [1]."):
        self.answer = answer
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.answer


@pytest.fixture
def retrieved(monkeypatch):
    """Stub search, returning as many chunks as the caller asked for."""
    seen = {}

    def fake_search(text, where=None, k=5):
        seen.update(text=text, where=where, k=k)
        return [(make_chunk(i), 0.9 - i / 100) for i in range(k)]

    monkeypatch.setattr(pipeline, "search", fake_search)
    return seen


def test_answer_carries_text_and_numbered_sources(retrieved):
    answer = pipeline.ask("Why is this raised?", "Monocyte", provider=FakeProvider())
    assert answer.panel.text.startswith("Monocytes rise")
    assert [s.number for s in answer.panel.sources] == [1, 2, 3, 4, 5]
    assert answer.panel.sources[0].title == "Monocytosis in practice"


def test_class_becomes_the_metadata_filter(retrieved):
    pipeline.ask("Why is this raised?", "Monocyte", provider=FakeProvider())
    assert retrieved["where"] == {
        "$or": [
            {"cell_classes": {"$contains": "Monocyte"}},
            {"cell_classes": {"$contains": "General"}},
        ]
    }


def test_the_users_wording_is_what_the_model_is_asked(retrieved):
    """The retrieval query is shaped for embedding similarity; prompting with it
    instead of the question is what made the panels abstain."""
    provider = FakeProvider()
    pipeline.ask("Why is this not a lymphocyte?", "Monocyte", provider=provider)
    assert "Why is this not a lymphocyte?" in provider.messages[-1]["content"]


def test_notable_findings_reach_the_retrieval_query(retrieved):
    result = {"cytoplasm_vacuole": ("yes", 0.99)}
    pipeline.ask("What does this mean?", "Monocyte", result, provider=FakeProvider())
    assert "vacuolation" in retrieved["text"]


def test_fabricated_citation_is_reported_not_raised(retrieved):
    answer = pipeline.ask(
        "Why is this raised?",
        "Monocyte",
        k=2,
        provider=FakeProvider("Sepsis is a cause [1][7]."),
    )
    assert answer.invalid == (7,)
    assert answer.panel.text  # still handed over, with the warning attached


def test_abstention_is_flagged(retrieved):
    answer = pipeline.ask(
        "What is the capital of France?", "Monocyte", provider=FakeProvider(ABSTAIN)
    )
    assert answer.abstained


def test_empty_retrieval_abstains_without_calling_the_provider(monkeypatch):
    monkeypatch.setattr(pipeline, "search", lambda text, where=None, k=5: [])
    provider = FakeProvider()
    answer = pipeline.ask("Anything?", "Monocyte", provider=provider)
    assert answer.abstained
    assert answer.panel.sources == []
    assert provider.messages is None
