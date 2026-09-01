"""Tests for prompt assembly and citation checking.

No network and no key: the prompt is built and the answer is inspected by pure
code, so the grounding rules are testable without spending anything.

The citation checks are the point. A model that cites a source it was never given
has invented something, and that has to be caught mechanically rather than
noticed by whoever happens to read the answer.
"""

import pytest

from claricyte.rag.corpus import Chunk
from claricyte.rag.generate import (
    ABSTAIN,
    SYSTEM_PROMPT,
    abstained,
    build_messages,
    cited,
    format_sources,
    invalid_citations,
    uncited_sentences,
)


def make_chunk(**overrides) -> Chunk:
    base = dict(
        text="Basophilia is associated with chronic myeloid leukaemia.",
        source_id="PMC1",
        section="Clinical associations",
        url="https://example.org/PMC1/",
        license="CC BY",
        cell_classes=("Basophil",),
        title="Basophils in health and disease",
        chunk_index=0,
    )
    base.update(overrides)
    return Chunk(**base)


SOURCES = [make_chunk(chunk_index=i) for i in range(3)]


# --- prompt ---------------------------------------------------------------


def test_sources_are_numbered_from_one():
    text = format_sources(SOURCES)
    assert text.startswith("[1]")
    assert "[3]" in text and "[4]" not in text
    assert "[0]" not in text


def test_sources_carry_title_and_section_for_attribution():
    assert "Basophils in health and disease" in format_sources(SOURCES)
    assert "Clinical associations" in format_sources(SOURCES)


def test_messages_put_rules_in_system_and_sources_in_user():
    messages = build_messages("why is this elevated", SOURCES)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "cite" in messages[0]["content"].lower()
    assert "why is this elevated" in messages[1]["content"]
    assert SOURCES[0].text in messages[1]["content"]


def test_prompt_forbids_describing_the_image():
    """The CBM owns what the cell looks like. If the LLM contradicts it the demo
    is showing two disagreeing accounts of one picture."""
    assert "not seen this cell" in SYSTEM_PROMPT
    # The distinction the rule turns on: the type may be described, this one
    # may not. Losing either half breaks it in a different direction.
    assert "TYPE" in SYSTEM_PROMPT
    assert "THIS one shows" in SYSTEM_PROMPT


def test_prompt_tells_the_model_what_to_do_with_conflicting_sources():
    """Clinical sources disagree often, and silently picking one hides that from
    a student who would be better served seeing the disagreement."""
    assert "disagree" in SYSTEM_PROMPT


def test_prompt_carries_the_exact_abstain_string():
    """The UI and the eval both match on it, so it cannot drift."""
    assert ABSTAIN in SYSTEM_PROMPT


# --- citation checking ----------------------------------------------------


def test_cited_finds_every_reference():
    assert cited("Seen in CML [1] and in allergy [2][3].") == {1, 2, 3}


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Both agree [1, 5].", {1, 5}),
        ("Compact form [1,5].", {1, 5}),
        ("Mixed [1,5] and [3].", {1, 3, 5}),
        ("Spaced [ 2 , 4 ].", {2, 4}),
    ],
)
def test_comma_grouped_citations_are_parsed(answer, expected):
    """Models write [1, 5] despite being asked for one bracket per source. Missing
    that form marked properly cited sentences as ungrounded, which would have made
    the groundedness metric meaningless."""
    assert cited(answer) == expected


def test_comma_grouped_citations_are_range_checked():
    assert invalid_citations("Claim [1, 9].", source_count=3) == {9}


def test_a_comma_grouped_citation_counts_as_cited():
    assert uncited_sentences("These conditions may lead to X [1, 5].") == []


def test_no_citations_is_an_empty_set():
    assert cited("Basophilia has several causes.") == set()


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Associated with CML [4].", {4}),
        ("Reported widely [0].", {0}),
        ("Both [1] and [9] agree.", {9}),
        ("Well established [1][2][3].", set()),
    ],
)
def test_citations_outside_the_supplied_range_are_flagged(answer, expected):
    """Three sources were given, so anything outside 1-3 was invented."""
    assert invalid_citations(answer, source_count=3) == expected


def test_abstention_is_recognised():
    assert abstained(ABSTAIN)
    assert abstained(f"  {ABSTAIN.upper()}  ")
    assert not abstained("Basophilia is associated with CML [1].")


def test_uncited_claims_are_flagged():
    answer = "Basophilia occurs in CML [1]. Eosinophils also rise in parasitic disease."
    assert uncited_sentences(answer) == ["Eosinophils also rise in parasitic disease."]


def test_a_fully_cited_answer_flags_nothing():
    answer = "Basophilia occurs in CML [1]. It is also seen in allergy [2]."
    assert uncited_sentences(answer) == []


def test_abstaining_is_not_an_uncited_claim():
    """Declining is the correct behaviour, not a grounding failure."""
    assert uncited_sentences(ABSTAIN) == []
