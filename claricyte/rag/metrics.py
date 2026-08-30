"""Retrieval and grounding metrics.

Pure arithmetic over ids the caller already has, so CI tests the scoring without
torch, a network or a key. The script that runs retrieval is plumbing around
this.

Scoring is at source level, not chunk level: the gold set names articles, and a
question is answered if any chunk of an accepted article comes back. Chunk-level
scoring would demand the gold set name chunk ids, which change on every rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_rank(retrieved: list[str], accepted: tuple[str, ...]) -> int | None:
    """1-based position of the first accepted source, or None if absent.

    Rank is the unit the other metrics are built from: hit rate throws away the
    position, MRR keeps it, and both need to agree about what counts as a hit.
    """
    wanted = set(accepted)
    for position, source_id in enumerate(retrieved, 1):
        if source_id in wanted:
            return position
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Fraction of questions with an accepted source in the top k.

    Ranks come from one retrieval at the largest k of interest, so hit rate at
    every smaller k is a filter over the same run rather than another retrieval.
    """
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank, counting a miss as zero.

    The difference from hit rate: retrieving the right article third scores 0.33
    here and a full 1.0 there. That matters because the generator reads all k
    chunks but weights the early ones, and because a low k would drop it.
    """
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def precision_at_k(retrieved: list[str], accepted: tuple[str, ...], k: int) -> float:
    """Fraction of the top k that are accepted sources.

    Reported alongside hit rate because they fail in opposite directions: hit
    rate rises with k for free, precision falls. A retriever that gets one right
    answer and four irrelevant ones scores 1.0 and 0.2, and both are true.
    """
    if k <= 0:
        return 0.0
    wanted = set(accepted)
    return sum(1 for source_id in retrieved[:k] if source_id in wanted) / k


@dataclass(frozen=True)
class RetrievalScore:
    """Scores for one configuration over one set of questions."""

    name: str
    questions: int
    ranks: list[int | None]
    precisions: list[float]

    def hit_rate(self, k: int) -> float:
        return hit_rate(self.ranks, k)

    @property
    def mrr(self) -> float:
        return mrr(self.ranks)

    @property
    def precision(self) -> float:
        if not self.precisions:
            return 0.0
        return sum(self.precisions) / len(self.precisions)

    @property
    def misses(self) -> int:
        return sum(1 for rank in self.ranks if rank is None)


def abstention_rate(abstained: list[bool]) -> float:
    """Fraction of answers that declined.

    Read against the adversarial subset, where it should be high, and against
    the answerable one, where it should be near zero. One number without the
    other says nothing: a system that abstains on everything scores perfectly
    on the first.
    """
    if not abstained:
        return 0.0
    return sum(abstained) / len(abstained)
