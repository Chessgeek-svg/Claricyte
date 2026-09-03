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
import sys
from dataclasses import replace
from typing import Callable, NamedTuple

from claricyte.rag.generate import (
    abstained,
    build_messages,
    invalid_citations,
    uncited_sentences,
)
from claricyte.rag.gold import GOLD_PATH, GoldQuestion, coverage, load_gold
from claricyte.rag.judge import (
    NO_CLAIM,
    UNSUPPORTED,
    agreement,
    groundedness,
    judge_answer,
    read_judgements,
    write_judgements,
    write_sources,
)
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
            # Titles, not just ids, because triage is judging whether what came
            # back answers the question, and an id says nothing about that.
            seen: dict[str, str] = {}
            for chunk, _ in retrieved:
                seen.setdefault(
                    chunk.source_id, f"{chunk.title[:52]} / {chunk.section[:22]}"
                )
            missed[question.id] = [f"{sid}  {title}" for sid, title in seen.items()]
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


JUDGEMENTS_PATH = "rag_data/eval/judgements.jsonl"
# The chunks behind the judgements. Without them a verdict is a bare number and
# nobody can check it, which makes the hand review the score depends on
# impossible.
SOURCES_PATH = "rag_data/eval/judged_sources.json"


def run_generation(
    questions: list[GoldQuestion],
    k: int,
    provider_name: str,
    judge: bool = False,
) -> dict:
    """Generate an answer per question and check it.

    Three of the four checks are mechanical: citations that resolve, sentences
    that carry one, and refusal when nothing supports an answer. The fourth is
    the judge, which is the only one that can catch a fluent invented sentence
    with a valid citation attached, and the only one whose own numbers need
    validating before they may be reported.
    """
    from claricyte.rag.store import search

    provider = get_provider(provider_name)
    answerable, adversarial = [], []
    fabricated, uncited, failures = 0, 0, []
    judgements, judged_chunks = [], {}

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
                failures.append(("answered, should have declined", question, answer))
        else:
            answerable.append(refused)
            if refused:
                failures.append(("declined, should have answered", question, answer))
        if bad:
            failures.append(
                (f"cited {sorted(bad)}, which do not exist", question, answer)
            )
        elif loose:
            failures.append((f"{len(loose)} uncited sentences", question, answer))

        # Only answers are judged. An abstention asserts nothing, so scoring it
        # would count a correct refusal as ungrounded.
        if judge and not refused:
            judgements.extend(judge_answer(answer, chunks, provider, question.id))
            judged_chunks[question.id] = chunks

    return {
        "judgements": judgements,
        "judged_chunks": judged_chunks,
        "adversarial_abstention": abstention_rate(adversarial),
        "answerable_abstention": abstention_rate(answerable),
        "fabricated_citations": fabricated,
        "uncited_sentences": uncited,
        "failures": failures,
    }


def report_judge(judgements: list, provider_name: str) -> dict:
    """Print the groundedness score, and say whether it may be quoted.

    Human verdicts already recorded are carried onto the new run, matched by
    question and sentence, so hand review survives a rerun of everything else.
    """
    previous = {
        (j.question_id, j.sentence): j.human for j in read_judgements(JUDGEMENTS_PATH)
    }
    for i, judgement in enumerate(judgements):
        human = previous.get((judgement.question_id, judgement.sentence))
        if human is not None:
            judgements[i] = replace(judgement, human=human)
    write_judgements(judgements, JUDGEMENTS_PATH)

    score = groundedness(judgements)
    claims = sum(1 for j in judgements if j.verdict != NO_CLAIM)
    rate, sample = agreement(judgements)

    print(f"\ngroundedness (judge: {provider_name})")
    print(f"  supported claims          {score:.2f} of {claims}")
    if sample:
        print(f"  judge/human agreement     {rate:.2f} over {sample} reviewed")
    else:
        print("  judge/human agreement     nothing reviewed yet, so the score above")
        print("                            is a lead, not a result. Fill in the")
        print(f'                            "human" field in {JUDGEMENTS_PATH}.')

    unsupported = [j for j in judgements if j.verdict == UNSUPPORTED]
    if unsupported:
        print(f"\n{len(unsupported)} statements judged unsupported:")
        for judgement in unsupported[:15]:
            sentence = " ".join(judgement.sentence.split())
            print(f"  {judgement.question_id:28} {sentence[:120]}")
    return {"groundedness": score, "claims": claims, "reviewed": sample}


def main() -> None:
    # Answers carry characters cp1252 cannot encode. Without this a Windows
    # console kills the run on the first print, after every API call is paid for.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="costs API calls")
    parser.add_argument(
        "--per-question", action="store_true", help="rank for every question"
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="score groundedness sentence by sentence; roughly doubles the cost",
    )
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

    by_id = {question.id: question for question in questions}
    if args.per_question:
        print("\nrank of the first accepted source, per question:")
        for question, rank in zip(scored, scores[0].ranks):
            mark = "miss" if rank is None else f"{rank:4}"
            print(f"  {mark}  {question.id:28} {question.question[:44]}")

    if misses[VARIANTS[0].name]:
        print(f"\n{len(misses[VARIANTS[0].name])} retrieval misses:")
        for question_id, got in misses[VARIANTS[0].name].items():
            question = by_id[question_id]
            print(f"\n  {question_id}: {question.question}")
            print(f"      wanted: {', '.join(question.sources)}")
            for line in got:
                print(f"      got:    {line}")
        print(
            "\n  Triage each: if something that came back does answer the question,\n"
            "  add it to that question's sources. If nothing does, it is a real\n"
            "  retrieval failure and the corpus or the query is at fault."
        )

    if args.generate:
        generation = run_generation(questions, args.k, args.provider, args.judge)
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
        if generation["failures"]:
            print(f"\n{len(generation['failures'])} generation findings:")
        for kind, question, answer in generation["failures"]:
            print(f"\n  {question.id}  [{kind}]")
            print(f"      asked:  {question.question}")
            print(f"      expect: {' '.join(question.expect.split())[:96]}")
            print(f"      got:    {' '.join(answer.split())[:280]}")
        if args.judge:
            write_sources(generation["judged_chunks"], SOURCES_PATH)
            results["generation_groundedness"] = report_judge(
                generation["judgements"], args.provider
            )
            print(f"  chunks behind them     -> {SOURCES_PATH}")

        results["generation"] = {
            key: value
            for key, value in generation.items()
            if key not in ("failures", "judgements", "judged_chunks")
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
