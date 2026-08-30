"""The evaluation gold set: questions, and what should come back for them.

Pure and network-free, like the rest of the package, so the shape of the eval
data is testable without running the eval.

Each question names every source that would be a correct retrieval, not the one
that happens to be best. Two articles often answer the same question, and a
metric that calls the second one a miss measures the gold set rather than the
retriever. The list is expected to grow: when a run misses, look at what came
back and either accept it or record a real failure.

Adversarial questions name no source. They are in the corpus's subject area but
outside what it covers, so the system should abstain. Their score is the
abstention rate, and it is the only direct measurement of the grounding rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from claricyte.vocab import CLASSES

GOLD_PATH = "rag_data/eval/gold.yaml"


@dataclass(frozen=True)
class GoldQuestion:
    """One evaluated question and its accepted retrievals."""

    id: str
    question: str
    # The cell on screen when the question is asked, which becomes the metadata
    # filter. Every question is asked about a cell, the same way the UI does.
    label: str
    # source_id values, any of which counts as a hit. Empty for adversarial.
    sources: tuple[str, ...] = ()
    adversarial: bool = False
    # One line on what a good answer says. For reading failures, not for scoring:
    # nothing compares generated text against it automatically.
    expect: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.label not in CLASSES:
            raise ValueError(f"{self.id}: unknown label {self.label!r}")
        if self.adversarial and self.sources:
            raise ValueError(
                f"{self.id}: adversarial questions name no source, since the "
                "corpus is not supposed to answer them"
            )
        if not self.adversarial and not self.sources:
            raise ValueError(
                f"{self.id}: no accepted sources. Name at least one, or mark it "
                "adversarial if nothing should answer it."
            )


def load_gold(path: str | Path = GOLD_PATH) -> list[GoldQuestion]:
    """Read the gold set, rejecting anything the eval could not score."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or []

    questions = [
        GoldQuestion(
            id=str(entry["id"]),
            question=entry["question"],
            label=entry["label"],
            sources=tuple(entry.get("sources", ())),
            adversarial=bool(entry.get("adversarial", False)),
            expect=entry.get("expect", ""),
            tags=tuple(entry.get("tags", ())),
        )
        for entry in raw
    ]

    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"duplicate question id {question.id!r}")
        seen.add(question.id)
    return questions


def unknown_sources(
    questions: list[GoldQuestion], corpus_source_ids: set[str]
) -> dict[str, tuple[str, ...]]:
    """Accepted sources that are not in the corpus, by question id.

    Separate from loading because it needs the corpus. A rebuild can drop an
    article, which silently turns a question into one nothing can ever hit;
    without this check that reads as a retrieval regression.
    """
    missing = {}
    for question in questions:
        absent = tuple(s for s in question.sources if s not in corpus_source_ids)
        if absent:
            missing[question.id] = absent
    return missing


def coverage(questions: list[GoldQuestion]) -> dict[str, int]:
    """Questions per class, plus the adversarial count.

    A gold set weighted towards one class measures that class. Printed before a
    run so the weighting is visible rather than assumed.
    """
    counts = {label: 0 for label in CLASSES}
    counts["adversarial"] = 0
    for question in questions:
        counts[question.label] += 1
        if question.adversarial:
            counts["adversarial"] += 1
    return counts
