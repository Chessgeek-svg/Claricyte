"""Tests for the groundedness judge.

Everything except the model call is pure. The cases that matter are the ones
where a wrong implementation still produces a plausible number:

  - a short or garbled reply silently padded to SUPPORTED would raise
    groundedness exactly when the judge is least reliable
  - counting NO_CLAIM sentences as supported rewards padding
  - agreement computed over unreviewed rows would report the judge agreeing
    with itself
"""

import pytest

from claricyte.rag.corpus import Chunk
from claricyte.rag.judge import (
    JUDGE_PROMPT,
    NO_CLAIM,
    SUPPORTED,
    UNSUPPORTED,
    Judgement,
    agreement,
    build_judge_messages,
    groundedness,
    judge_answer,
    parse_verdicts,
    read_judgements,
    write_judgements,
)


def make_chunk(index: int = 0) -> Chunk:
    return Chunk(
        text="Basophilia is a feature of chronic myeloid leukaemia.",
        source_id="PMC1",
        section="Discussion",
        url="https://example.org",
        license="CC BY",
        cell_classes=("Basophil",),
        title="Basophils in CML",
        chunk_index=index,
    )


class FakeProvider:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.reply


# --- the prompt -----------------------------------------------------------


def test_prompt_separates_support_from_truth():
    """The whole point. "Segmentation aids migration" is true and unsupported,
    and a judge scoring truth would pass it."""
    assert "NOT checking whether a statement is true" in JUDGE_PROMPT
    assert "can be correct" in JUDGE_PROMPT


def test_sources_are_sent_once_for_all_sentences():
    messages = build_judge_messages(["One.", "Two.", "Three."], [make_chunk()])
    content = messages[-1]["content"]
    assert content.count("Basophilia is a feature") == 1
    assert "1. One." in content and "3. Three." in content


# --- parsing --------------------------------------------------------------


def test_verdicts_come_back_in_statement_order():
    reply = f"2: {UNSUPPORTED}\n1: {SUPPORTED}"
    assert parse_verdicts(reply, 2) == [SUPPORTED, UNSUPPORTED]


@pytest.mark.parametrize(
    "line", ["1: {v}", "1. {v}", "  1)  {v}", "1: {v} (excerpt 2)"]
)
def test_verdict_lines_are_parsed_in_the_forms_models_emit(line):
    assert parse_verdicts(line.format(v=SUPPORTED), 1) == [SUPPORTED]


def test_a_missing_verdict_raises_rather_than_defaulting():
    """Padding to SUPPORTED would inflate groundedness precisely when the judge
    is struggling."""
    with pytest.raises(ValueError, match="no verdict for statement"):
        parse_verdicts(f"1: {SUPPORTED}", expected=3)


def test_an_unparseable_reply_raises():
    with pytest.raises(ValueError):
        parse_verdicts("Looks fine to me.", expected=1)


# --- scoring --------------------------------------------------------------


def judged(*verdicts, human=None):
    return [Judgement("q", f"sentence {i}", v, human) for i, v in enumerate(verdicts)]


def test_groundedness_is_the_supported_share():
    assert groundedness(judged(SUPPORTED, SUPPORTED, UNSUPPORTED)) == pytest.approx(
        2 / 3
    )


def test_sentences_asserting_nothing_leave_the_denominator():
    """Otherwise a wordier answer outscores a terse one saying the same thing."""
    assert groundedness(judged(SUPPORTED, UNSUPPORTED, NO_CLAIM)) == 0.5


def test_an_answer_of_pure_transitions_scores_zero_not_one():
    assert groundedness(judged(NO_CLAIM, NO_CLAIM)) == 0.0


def test_groundedness_of_nothing_is_zero():
    assert groundedness([]) == 0.0


# --- agreement ------------------------------------------------------------


def test_agreement_ignores_unreviewed_rows():
    """A judge agreeing with itself is not validation."""
    rows = judged(SUPPORTED, SUPPORTED) + judged(UNSUPPORTED, human=UNSUPPORTED)
    rate, sample = agreement(rows)
    assert (rate, sample) == (1.0, 1)


def test_agreement_counts_disagreement():
    rows = judged(SUPPORTED, human=UNSUPPORTED) + judged(SUPPORTED, human=SUPPORTED)
    assert agreement(rows) == (0.5, 2)


def test_no_review_yet_reports_a_sample_of_zero():
    assert agreement(judged(SUPPORTED)) == (0.0, 0)


# --- calling the judge ----------------------------------------------------


def test_judge_answer_splits_and_labels_each_sentence():
    provider = FakeProvider(f"1: {SUPPORTED}\n2: {UNSUPPORTED}")
    result = judge_answer(
        "Basophilia occurs in CML [1]. It also aids migration [1].",
        [make_chunk()],
        provider,
        question_id="q1",
    )
    assert [j.verdict for j in result] == [SUPPORTED, UNSUPPORTED]
    assert all(j.question_id == "q1" for j in result)


def test_decimals_do_not_split_a_sentence():
    """A naive split on ". " would cut "1.5 x 10^9/L" in half and desynchronise
    every verdict after it."""
    provider = FakeProvider(f"1: {SUPPORTED}")
    result = judge_answer(
        "Hypereosinophilia is 1.5 x 10^9/L or more [1].", [make_chunk()], provider
    )
    assert len(result) == 1


def test_an_empty_answer_calls_nothing():
    provider = FakeProvider("unused")
    assert judge_answer("", [make_chunk()], provider) == []
    assert provider.messages is None


# --- persistence ----------------------------------------------------------


def test_judgements_round_trip(tmp_path):
    path = tmp_path / "judgements.jsonl"
    write_judgements(judged(SUPPORTED, UNSUPPORTED), path)
    assert [j.verdict for j in read_judgements(path)] == [SUPPORTED, UNSUPPORTED]


def test_rerunning_the_judge_preserves_human_verdicts(tmp_path):
    """Hand review is the expensive part. A rerun must not discard it."""
    path = tmp_path / "judgements.jsonl"
    write_judgements([Judgement("q", "A claim.", SUPPORTED, UNSUPPORTED)], path)
    write_judgements([Judgement("q", "A claim.", SUPPORTED)], path)
    assert read_judgements(path)[0].human == UNSUPPORTED


def test_reading_a_missing_file_is_not_an_error(tmp_path):
    assert read_judgements(tmp_path / "absent.jsonl") == []
