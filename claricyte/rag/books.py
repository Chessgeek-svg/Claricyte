"""Open-licensed textbooks from LibreTexts.

A second front door onto the same Article shape the PMC path produces, so
everything downstream is unchanged.

These fill the gap the journal corpus has: descriptive morphology and the
band-versus-seg criteria, which clinical reviews assume you already know.

Scraped rather than fetched through an API, because LibreTexts returns 403 on
its deki endpoints while serving the HTML fine. Content lives in a
``mt-content-container`` section; child pages are links inside it.
"""

from __future__ import annotations

import html
import re
import time
import urllib.request
from dataclasses import dataclass

from claricyte.rag.pmc import Article

USER_AGENT = "claricyte-corpus/0.1 (educational research; open-licensed content)"
REQUEST_INTERVAL_SECONDS = 0.5

# Below this a page is a stub: a heading, a licence footer, and nothing to
# retrieve. Most of the Lab Guide is exactly that, its content being images.
MIN_WORDS = 60

CONTENT = re.compile(
    r'<section class="mt-content-container[^"]*"[^>]*>(.*?)</section>', re.S
)
CHILD_LINK = re.compile(
    r'<a[^>]+href="(https://med\.libretexts\.org/[^"]+)"[^>]*>(.*?)</a>', re.S
)
SCRIPTS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
TAGS = re.compile(r"<[^>]+>")
# LibreTexts appends a licence and provenance footer to every page.
FOOTER = re.compile(r"This page titled.*$", re.S)
# LibreTexts numbers figures with a LaTeX macro that survives tag stripping.
PAGEINDEX = re.compile(r"\\\(\\PageIndex\{[^}]*\}\\\)")


@dataclass(frozen=True)
class Book:
    """One open-licensed book and the parts of it worth retrieving."""

    source_id: str
    title: str
    url: str
    license: str
    # Page-name prefixes to descend into. Everything else (red cells, leukaemias,
    # front matter) is out of scope for the current six classes.
    include: tuple[str, ...]


BOOKS: tuple[Book, ...] = (
    Book(
        source_id="OER-ATLAS",
        title=(
            "Clinical Hematology Atlas: A Pictorial Guide for the Hematology Laboratory"
        ),
        url=(
            "https://med.libretexts.org/Courses/Oregon_Institute_of_Technology/"
            "Clinical_Hematology_Atlas%3A_A_Pictorial_Guide_for_the_Hematology_"
            "Laboratory_(Taylor_and_Doty)"
        ),
        license="CC BY-NC-SA",
        include=(
            "Normal Blood Cells",
            "White Blood Cell Variants",
            "Slide and Stain Quality",
            "Other Abnormal Cells",
        ),
    ),
    Book(
        source_id="OER-LABGUIDE",
        title="A Laboratory Guide to Clinical Hematology",
        url=(
            "https://med.libretexts.org/Bookshelves/Allied_Health/"
            "A_Laboratory_Guide_to_Clinical_Hematology_(Villatoro_and_To)"
        ),
        license="CC BY-NC",
        include=(
            "10: White Blood Cells and Platelets",
            "11: White Blood Cells- Non-Malignant",
        ),
    ),
)


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", "ignore")
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return body


def _content(page: str) -> str:
    match = CONTENT.search(page)
    return match.group(1) if match else ""


def _children(content: str) -> list[tuple[str, str]]:
    """(name, url) of the child pages linked from a page's content."""
    seen: set[str] = set()
    children = []
    for url, label in CHILD_LINK.findall(content):
        name = " ".join(TAGS.sub(" ", label).replace("&nbsp;", " ").split())
        if name and url not in seen:
            seen.add(url)
            children.append((name, url))
    return children


def _text(content: str) -> str:
    """Readable text, minus scripts, LaTeX markers and the licence footer."""
    stripped = TAGS.sub(" ", SCRIPTS.sub(" ", content))
    unescaped = html.unescape(stripped)
    # LibreTexts numbers its figures with a LaTeX macro that survives tag
    # stripping and would otherwise be embedded as if it were prose.
    return FOOTER.sub("", " ".join(PAGEINDEX.sub(" ", unescaped).split())).strip()


def load_book(book: Book, min_words: int = MIN_WORDS) -> Article:
    """Fetch a book's included pages as one Article of (heading, text) sections.

    Descends one level into each included page, since LibreTexts nests content
    under chapter pages that are themselves only tables of contents.
    """
    sections: list[tuple[str, str]] = []
    for name, url in _children(_content(_get(book.url))):
        if not name.startswith(book.include):
            continue
        content = _content(_get(url))
        for heading, text in _pages(name, content):
            if len(text.split()) >= min_words:
                sections.append((heading, text))
    return Article(
        source_id=book.source_id,
        title=book.title,
        url=book.url,
        license=book.license,
        sections=tuple(sections),
    )


def _pages(name: str, content: str) -> list[tuple[str, str]]:
    """A page's own text plus its children's, as (heading, text)."""
    found = [(name, _text(content))]
    for child_name, child_url in _children(content):
        if child_name == name:
            continue  # LibreTexts links a page to itself in its own listing
        found.append((child_name, _text(_content(_get(child_url)))))
    return found
