"""Tests for textbook chapter selection.

Only Book.wants is testable without network, and it is the part that decides
what the corpus contains. The rest of books.py is scraping.
"""

import pytest

from claricyte.rag.books import BOOKS, Book


def make_book(**overrides) -> Book:
    base = dict(
        source_id="OER-TEST",
        title="A Guide to Something",
        url="https://med.libretexts.org/x",
        license="CC BY-NC",
    )
    base.update(overrides)
    return Book(**base)


def test_empty_include_takes_the_whole_book():
    book = make_book()
    assert book.wants("1: Red Blood Cells")
    assert book.wants("13: Mature Lymphoid Neoplasms")


def test_include_restricts_to_matching_prefixes():
    book = make_book(include=("10: White Blood Cells",))
    assert book.wants("10: White Blood Cells and Platelets")
    assert not book.wants("1: Red Blood Cells")


@pytest.mark.parametrize("name", ["Front Matter", "Back Matter", "Resources"])
def test_front_and_back_matter_are_skipped(name):
    assert not make_book().wants(name)


def test_skipped_even_when_include_would_match():
    """A prefix broad enough to catch the front matter must not resurrect it."""
    assert not make_book(include=("Front",)).wants("Front Matter")


def test_the_books_self_link_is_skipped():
    """LibreTexts lists every book inside its own table of contents, which would
    otherwise be fetched again as a chapter of itself."""
    book = make_book()
    assert not book.wants(book.title)


def test_shipped_books_take_everything():
    """Both are wanted in full: a chapter on a cell the model cannot yet predict
    still answers questions about that cell, and untagged chunks are dropped."""
    assert all(book.include == () for book in BOOKS)
