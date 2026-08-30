"""Evaluate retrieval and grounding against the gold set.

    python scripts/eval_rag.py                  retrieval only, free
    python scripts/eval_rag.py --generate       adds the generation checks

Retrieval is scored under four configurations so the design decisions carry
numbers rather than arguments. The one that matters is the class filter: it is
the project's central claim, and "no class filter" is its baseline.

Generation is behind a flag because it costs money. Retrieval is free and is
what changes when chunking, k or the query form change, so it should be
runnable constantly.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, NamedTuple

from claricyte.rag.generate import (
    abstained,
    build_messages,
    invalid_citations,
    uncited_sentences,
)
from claricyte.rag.gold import GOLD_PATH, GoldQuestion, coverage, load_gold
from claricyte.rag.metrics import (
    RetrievalScore,
    abstention_rate,
    hit_rank,
    precision_at_k,
)
from claricyte.rag.providers import get_provider
from claricyte.rag.query import build_query, class_filter

# Hit rate is reported at each of these, derived from one retrieval at --k
# rather than from one run per k, since the top 3 of a top 5 run is the top 3.
REPORT_AT = (1, 3, 5, 8)


class Variant(NamedTuple):
    """One retrieval configuration to score."""

    name: str
    text_of: Callable[[GoldQuestion], str]
    use_filter: bool
    note: str


def _prose(question: GoldQuestion) -> str:
    """What the app actually sends: the class prepended to the user's wording."""
    text, _ = build_query({}, question.label, question=question.question)
    return text


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "full",
        _prose,
        True,
        "what the app does: class as metadata filter, question as prose query",
    ),
    Variant(
        "no class filter",
        _prose,
        False,
        "the baseline for the central claim. Same query, whole corpus.",
    ),
    Variant(
        "no class prefix",
        lambda question: question.question,
        True,
        "does prepending the class to the query text earn its place?",
    ),
    Variant(
        "class name only",
        lambda question: f"{question.label.lower()}s in peripheral blood",
        True,
        "does the question's wording beat naming the class? If not, "
        "semantic search is doing nothing the filter was not already doing.",
    ),
)


def score_retrieval(
    variant: Variant, questions: list[GoldQuestion], k: int
) -> tuple[RetrievalScore, dict[str, list[str]]]:
    """Score one configuration, and record what came back for the misses."""
    from claricyte.rag.store import search

    ranks: list[int | None] = []
    precisions: list[float] = []
    missed: dict[str, list[str]] = {}
    for question in questions:
        where = class_filter(question.label) if variant.use_filter else None
        retrieved = search(variant.text_of(question), where=where, k=k)
        sources = [chunk.source_id for chunk, _ in retrieved]
        rank = hit_rank(sources, question.sources)
        ranks.append(rank)
        precisions.append(precision_at_k(sources, question.sources, k))
        if rank is None:
            # Kept so a miss can be triaged: either the retriever failed, or the
            # gold set is missing a source that also answers the question.
            missed[question.id] = sources
    return RetrievalScore(variant.name, len(questions), ranks, precisions), missed


def print_retrieval(scores: list[RetrievalScore], k: int) -> None:
    at = [n for n in REPORT_AT if n <= k]
    header = "".join(f"  hit@{n}" for n in at)
    print(f"\n{'configuration':22}{header}     MRR    P@k  misses")
    for score in scores:
        cells = "".join(f"  {score.hit_rate(n):5.2f}" for n in at)
        print(
            f"{score.name:22}{cells}  {score.mrr:5.2f}  "
            f"{score.precision:5.2f}  {score.misses:5}"
        )


def run_generation(questions: list[GoldQuestion], k: int, provider_name: str) -> dict:
    """Generate an answer per question and check it mechanically.

    No LLM judge. A sentence-level judge is the standard way to score
    groundedness, but an unvalidated judge is not a measurement, and validating
    one means hand-labelling a sample and reporting the agreement rate. Until
    that exists these three checks are what can be claimed honestly: citations
    that resolve, sentences that carry one, and refusal when nothing supports an
    answer.
    """
    from claricyte.rag.store import search

    provider = get_provider(provider_name)
    answerable, adversarial = [], []
    fabricated, uncited, failures = 0, 0, []

    for question in questions:
        text, where = build_query({}, question.label, question=question.question)
        chunks = [chunk for chunk, _ in search(text, where=where, k=k)]
        answer = provider.generate(build_messages(question.question, chunks))

        refused = abstained(answer)
        bad = invalid_citations(answer, len(chunks))
        loose = uncited_sentences(answer)
        fabricated += len(bad)
        uncited += len(loose)

        if question.adversarial:
            adversarial.append(refused)
            if not refused:
                failures.append((question.id, answer))
        else:
            answerable.append(refused)
            if refused:
                failures.append((question.id, answer))

    return {
        "adversarial_abstention": abstention_rate(adversarial),
        "answerable_abstention": abstention_rate(answerable),
        "fabricated_citations": fabricated,
        "uncited_sentences": uncited,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="costs API calls")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--out", help="write results as json")
    args = parser.parse_args()

    questions = load_gold(args.gold)
    scored = [q for q in questions if not q.adversarial]

    print(f"{len(questions)} questions ({len(scored)} scored on retrieval)")
    for label, count in coverage(questions).items():
        print(f"  {label:22} {count}")
    if not scored:
        raise SystemExit("nothing to score: every question is adversarial")

    results: dict = {"k": args.k, "questions": len(questions)}
    scores, misses = [], {}
    for variant in VARIANTS:
        score, missed = score_retrieval(variant, scored, args.k)
        scores.append(score)
        misses[variant.name] = missed
        results[variant.name] = {
            "hit_rate": {n: score.hit_rate(n) for n in REPORT_AT if n <= args.k},
            "mrr": score.mrr,
            "precision": score.precision,
        }

    print_retrieval(scores, args.k)
    print()
    for variant in VARIANTS:
        print(f"  {variant.name:22} {variant.note}")

    if misses[VARIANTS[0].name]:
        print("\nmisses under the shipped configuration:")
        for question_id, sources in misses[VARIANTS[0].name].items():
            print(f"  {question_id}")
            print(f"      got: {', '.join(dict.fromkeys(sources))}")
        print(
            "\n  Triage each: if one of those answers the question, add it to the\n"
            "  question's sources. If none does, it is a real retrieval failure."
        )

    if args.generate:
        generation = run_generation(questions, args.k, args.provider)
        print(
            f"\ngeneration ({args.provider})\n"
            f"  abstention, adversarial   {generation['adversarial_abstention']:.2f}"
            "   (want high)\n"
            f"  abstention, answerable    {generation['answerable_abstention']:.2f}"
            "   (want low)\n"
            f"  fabricated citations      {generation['fabricated_citations']}"
            "      (want 0)\n"
            f"  uncited sentences         {generation['uncited_sentences']}"
        )
        for question_id, answer in generation["failures"]:
            print(f"\n  {question_id}: {answer[:300]}")
        results["generation"] = {
            key: value for key, value in generation.items() if key != "failures"
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
