"""Sentence-level groundedness judging.

The mechanical checks in generate answer "do the citations resolve" and "does
every sentence carry one". Neither answers "is this sentence actually in the
sources", and the gap between them is where the interesting failure lives: asked
why a neutrophil nucleus is segmented, the model replied that segmentation aids
deformability and migration and cited two chunks, neither of which says anything
of the kind. Valid citations, every sentence bracketed, entirely invented.

The judge is asked one question and it is deliberately not "is this true". A
sentence can be correct and unsupported, and that is exactly the case to catch,
so the prompt draws the distinction in as many words.

Prompt building and parsing are pure and tested. Only judge_answer calls out.

An unvalidated judge is not a measurement. Judgements are written to disk with a
blank field for a human verdict, and agreement over a hand-checked sample is
what makes the groundedness number reportable. Until that sample exists the
number is a signal for finding failures, not a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from claricyte.rag.corpus import Chunk, split_sentences

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
NO_CLAIM = "NO_CLAIM"
VERDICTS = (SUPPORTED, UNSUPPORTED, NO_CLAIM)

JUDGE_PROMPT = f"""You check whether statements are supported by numbered source \
excerpts.

You are NOT checking whether a statement is true. A statement can be correct, \
widely known, and still unsupported here. If the excerpts do not say it, it is \
{UNSUPPORTED}, however plausible it sounds.

For each numbered statement reply on its own line, in the form "N: VERDICT", \
using exactly one of:
  {SUPPORTED}    the excerpts state this, or state something it follows directly from
  {UNSUPPORTED}  the excerpts do not state this
  {NO_CLAIM}     the statement asserts nothing checkable, such as a transition

Output only those lines. No preamble, no explanation, no blank lines.
"""

_VERDICT_LINE = re.compile(
    r"^\s*(\d+)\s*[:.)]\s*(" + "|".join(VERDICTS) + r")\b", re.MULTILINE
)


@dataclass(frozen=True)
class Judgement:
    """One judged sentence, with room for a human to disagree."""

    question_id: str
    sentence: str
    verdict: str
    # Filled in by hand during validation. None means not yet reviewed, which is
    # different from reviewed and agreed.
    human: str | None = None


def build_judge_messages(sentences: list[str], chunks: list[Chunk]) -> list[dict]:
    """Messages judging every sentence of one answer against its sources.

    One call per answer, not per sentence: the excerpts are the bulk of the
    tokens and resending them for each sentence would multiply the cost by the
    sentence count for no gain.
    """
    excerpts = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, 1))
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"Excerpts:\n{excerpts}\n\nStatements:\n{numbered}",
        },
    ]


def parse_verdicts(reply: str, expected: int) -> list[str]:
    """Verdicts in statement order.

    Raises rather than padding a short reply. A missing verdict silently read as
    SUPPORTED would bias groundedness upwards exactly when the judge is
    struggling, which is when the number most needs to be trusted.
    """
    found = {int(n): verdict for n, verdict in _VERDICT_LINE.findall(reply)}
    missing = [i for i in range(1, expected + 1) if i not in found]
    if missing:
        raise ValueError(
            f"judge returned no verdict for statement(s) {missing} "
            f"of {expected}. Reply was: {reply[:200]!r}"
        )
    return [found[i] for i in range(1, expected + 1)]


def judge_answer(
    answer: str, chunks: list[Chunk], provider, question_id: str = ""
) -> list[Judgement]:
    """Judge each sentence of an answer against the chunks it was given."""
    sentences = [s for s in split_sentences(answer) if s]
    if not sentences:
        return []
    reply = provider.generate(build_judge_messages(sentences, chunks))
    verdicts = parse_verdicts(reply, len(sentences))
    return [
        Judgement(question_id=question_id, sentence=sentence, verdict=verdict)
        for sentence, verdict in zip(sentences, verdicts)
    ]


def groundedness(judgements: list[Judgement]) -> float:
    """Supported share of the sentences that assert anything.

    NO_CLAIM sentences leave the denominator: counting a transition as supported
    would let a wordier answer score better than a terse one saying the same
    thing.
    """
    claims = [j for j in judgements if j.verdict != NO_CLAIM]
    if not claims:
        return 0.0
    return sum(1 for j in claims if j.verdict == SUPPORTED) / len(claims)


def agreement(judgements: list[Judgement]) -> tuple[float, int]:
    """(agreement rate, sample size) over the hand-reviewed judgements.

    This is the number that decides whether groundedness above may be reported
    at all. Reviewing a sample and finding the judge agrees 60% of the time
    means the groundedness figure is the judge's opinion, not a measurement.
    """
    reviewed = [j for j in judgements if j.human is not None]
    if not reviewed:
        return 0.0, 0
    agreed = sum(1 for j in reviewed if j.human == j.verdict)
    return agreed / len(reviewed), len(reviewed)


def write_judgements(judgements: list[Judgement], path: str | Path) -> None:
    """Write judgements for hand review, preserving any human verdicts already
    recorded against the same question and sentence."""
    path = Path(path)
    existing = {(j.question_id, j.sentence): j.human for j in read_judgements(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for judgement in judgements:
            human = existing.get((judgement.question_id, judgement.sentence))
            handle.write(
                json.dumps(
                    {
                        "question_id": judgement.question_id,
                        "sentence": judgement.sentence,
                        "verdict": judgement.verdict,
                        "human": human if human is not None else judgement.human,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_sources(sources: dict[str, list[Chunk]], path: str | Path) -> None:
    """Write the numbered chunks each question was answered from.

    Written beside the judgements because a verdict is unreviewable without
    them: "sentence 2 is unsupported, it cites [2]" is not something a person
    can check against a bare number. Kept in its own file rather than repeated
    on every judgement, since one question's five chunks back all of its
    sentences.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        question_id: [
            {
                "number": i,
                "source_id": chunk.source_id,
                "section": chunk.section,
                "url": chunk.url,
                "text": chunk.text,
            }
            for i, chunk in enumerate(chunks, 1)
        ]
        for question_id, chunks in sources.items()
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_judgements(path: str | Path) -> list[Judgement]:
    """Read judgements back. Missing file means none yet, not an error."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [Judgement(**json.loads(line)) for line in handle if line.strip()]
