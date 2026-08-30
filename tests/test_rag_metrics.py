"""Tests for the retrieval metrics.

These are the numbers the project's central claim will be argued with, so the
cases that matter are the ones where a plausible implementation is quietly
wrong: a miss counted as rank 0 instead of no rank, hit rate at k reading past
k, MRR averaging over hits rather than over questions.
"""

import pytest

from claricyte.rag.metrics import (
    RetrievalScore,
    abstention_rate,
    hit_rank,
    hit_rate,
    mrr,
    precision_at_k,
)

# --- hit_rank -------------------------------------------------------------


def test_rank_is_one_based():
    """Rank 1, not 0: MRR divides by it."""
    assert hit_rank(["PMC1", "PMC2"], ("PMC1",)) == 1


def test_rank_finds_the_first_accepted_source():
    assert hit_rank(["PMC9", "PMC2", "PMC1"], ("PMC1", "PMC2")) == 2


def test_missing_source_is_none_not_zero():
    assert hit_rank(["PMC9"], ("PMC1",)) is None


def test_repeated_chunks_from_one_article_do_not_change_the_rank():
    """Retrieval returns chunks; several can share a source. The rank is the
    position of the first, not the count."""
    assert hit_rank(["PMC9", "PMC1", "PMC1"], ("PMC1",)) == 2


def test_empty_retrieval_misses():
    assert hit_rank([], ("PMC1",)) is None


# --- hit_rate -------------------------------------------------------------


def test_hit_rate_counts_only_ranks_within_k():
    assert hit_rate([1, 4], k=3) == 0.5


def test_hit_rate_treats_a_miss_as_a_failure():
    assert hit_rate([1, None], k=5) == 0.5


def test_hit_rate_is_monotonic_in_k():
    ranks = [1, 3, 5, None]
    assert hit_rate(ranks, 1) <= hit_rate(ranks, 3) <= hit_rate(ranks, 5)


def test_hit_rate_of_nothing_is_zero_not_an_error():
    assert hit_rate([], k=5) == 0.0


# --- mrr ------------------------------------------------------------------


def test_mrr_is_the_reciprocal_of_the_rank():
    assert mrr([2]) == 0.5


def test_mrr_averages_over_questions_including_misses():
    """The trap: averaging over hits only turns a retriever that misses half the
    gold set into a perfect one."""
    assert mrr([1, None]) == 0.5


def test_mrr_separates_configurations_that_hit_rate_ties():
    """Both find the answer within 5, so hit rate at 5 cannot tell them apart."""
    first, third = [1, 1], [3, 3]
    assert hit_rate(first, 5) == hit_rate(third, 5)
    assert mrr(first) > mrr(third)


# --- precision ------------------------------------------------------------


def test_precision_counts_accepted_sources_in_the_top_k():
    assert precision_at_k(["PMC1", "PMC9", "PMC1", "PMC8"], ("PMC1",), k=4) == 0.5


def test_precision_divides_by_k_not_by_what_came_back():
    """Asking for 5 and getting 1 right out of 2 returned is 0.2, not 0.5:
    the empty slots are a real cost to the generator's context."""
    assert precision_at_k(["PMC1", "PMC9"], ("PMC1",), k=5) == pytest.approx(0.2)


def test_precision_and_hit_rate_disagree_by_design():
    retrieved = ["PMC9", "PMC1", "PMC9", "PMC9", "PMC9"]
    assert hit_rate([hit_rank(retrieved, ("PMC1",))], k=5) == 1.0
    assert precision_at_k(retrieved, ("PMC1",), k=5) == pytest.approx(0.2)


# --- RetrievalScore -------------------------------------------------------


def test_score_reports_misses_and_derived_metrics():
    score = RetrievalScore("full", 3, [1, 3, None], [0.2, 0.2, 0.0])
    assert score.misses == 1
    assert score.hit_rate(1) == pytest.approx(1 / 3)
    assert score.hit_rate(3) == pytest.approx(2 / 3)
    assert score.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)


# --- abstention -----------------------------------------------------------


def test_abstention_rate_is_a_fraction_of_answers():
    assert abstention_rate([True, True, False, False]) == 0.5


def test_abstention_rate_of_nothing_is_zero():
    assert abstention_rate([]) == 0.0
