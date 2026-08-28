"""Tests for JATS parsing and the licence filter.

These run against fixture XML rather than the network, so CI needs no NCBI access.
The fixtures mirror shapes seen in real PMC responses, including the two that broke
the parser when it was first run against live data: the licence declared as
``ali:license_ref`` element text rather than an attribute, and ``article-id`` using
``pub-id-type="pmcid"`` rather than ``"pmc"``.

The licence filter is the part worth testing hardest. A false positive here puts
text we may not redistribute into a committed corpus, and nothing downstream would
ever notice.
"""

import xml.etree.ElementTree as ET

import pytest

from claricyte.rag.pmc import (
    TEXT_MINING_LICENCE,
    _licence_of,
    _text_of,
    parse_article,
)

CC_URL = "https://creativecommons.org/licenses/{code}/4.0/"
TEXT_MINING = (
    "This file is available for text mining. It may also be used consistent "
    "with the principles of fair use under the copyright law."
)


def licence_block(*, url: str | None = None, text: str | None = None) -> str:
    """A JATS <license> in the ali:license_ref form PMC actually emits."""
    inner = ""
    if url is not None:
        inner += f'<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">{url}</ali:license_ref>'
    if text is not None:
        inner += f"<license-p>{text}</license-p>"
    return f"<license>{inner}</license>"


def article_xml(licence: str = "", body: str = "", pmcid: str = "PMC1") -> bytes:
    """A minimal but structurally real JATS article."""
    return f"""<article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmcid">{pmcid}</article-id>
          <title-group><article-title>A Title</article-title></title-group>
          <permissions>{licence}</permissions>
        </article-meta>
      </front>
      <body>{body}</body>
    </article>""".encode()


def section(title: str, text: str, sec_type: str = "") -> str:
    attr = f' sec-type="{sec_type}"' if sec_type else ""
    return f"<sec{attr}><title>{title}</title><p>{text}</p></sec>"


# --- licence filter -------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("by", "CC BY"),
        ("by-sa", "CC BY-SA"),
        ("by-nc", "CC BY-NC"),
        ("by-nc-sa", "CC BY-NC-SA"),
    ],
)
def test_permitted_licences_are_recognised(code, expected):
    root = ET.fromstring(article_xml(licence_block(url=CC_URL.format(code=code))))
    assert _licence_of(root) == expected


@pytest.mark.parametrize("code", ["by-nd", "by-nc-nd"])
def test_nodervatives_licences_are_rejected(code):
    """ND conflicts with chunking and reassembling, which is the whole pipeline."""
    root = ET.fromstring(article_xml(licence_block(url=CC_URL.format(code=code))))
    assert _licence_of(root) is None


def test_text_mining_terms_are_recognised_as_their_own_value():
    """Recorded distinctly, not flattened into a CC licence, so it stays filterable."""
    root = ET.fromstring(article_xml(licence_block(text=TEXT_MINING)))
    assert _licence_of(root) == TEXT_MINING_LICENCE


def test_article_with_no_licence_element_is_rejected():
    assert _licence_of(ET.fromstring(article_xml())) is None


def test_licence_is_found_in_element_text_not_only_attributes():
    """The bug found against live data: publishers put the URL in ali:license_ref."""
    root = ET.fromstring(article_xml(licence_block(url=CC_URL.format(code="by"))))
    assert root.find(".//license").get("license-type") is None
    assert _licence_of(root) == "CC BY"


# --- article parsing ------------------------------------------------------


def test_parses_sections_in_document_order():
    xml = article_xml(
        licence_block(url=CC_URL.format(code="by")),
        section("Introduction", "First paragraph.") + section("Results", "Second."),
    )
    article = parse_article(xml)
    assert [heading for heading, _ in article.sections] == ["Introduction", "Results"]
    assert article.pmcid == "PMC1"
    assert article.license == "CC BY"
    assert article.url.endswith("/PMC1/")


def test_article_without_a_body_is_dropped():
    """efetch answers for non-OA articles with an abstract and no body rather than
    failing, so this has to be checked or the corpus fills with stubs."""
    xml = article_xml(licence_block(url=CC_URL.format(code="by")))
    assert parse_article(xml) is None


def test_article_with_rejected_licence_is_dropped():
    xml = article_xml(
        licence_block(url=CC_URL.format(code="by-nc-nd")),
        section("Introduction", "Text."),
    )
    assert parse_article(xml) is None


@pytest.mark.parametrize(
    "heading", ["References", "Funding", "Acknowledgements", "Conflict of Interest"]
)
def test_boilerplate_sections_are_skipped(heading):
    """A reference list would flood the index with author names and journal titles."""
    xml = article_xml(
        licence_block(url=CC_URL.format(code="by")),
        section("Results", "Real content.") + section(heading, "Boilerplate."),
    )
    assert [h for h, _ in parse_article(xml).sections] == ["Results"]


def test_article_of_only_boilerplate_is_dropped():
    xml = article_xml(
        licence_block(url=CC_URL.format(code="by")),
        section("References", "1. Smith J."),
    )
    assert parse_article(xml) is None


def test_citation_markers_are_stripped_from_text():
    """An inline [12] adds no meaning and only pollutes the embedding."""
    element = ET.fromstring("<p>Neutrophils are common <xref>[12]</xref> in blood.</p>")
    assert "12" not in _text_of(element)
    assert _text_of(element) == "Neutrophils are common in blood."
