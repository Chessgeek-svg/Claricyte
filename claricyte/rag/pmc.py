"""PubMed Central fetching and JATS XML parsing.

Turns a search query into parsed, license-cleared articles ready to be chunked.
Everything network- and XML-facing lives here so the rest of the RAG package stays
pure and testable, the same split ``claricyte.explain`` uses.

Each stage is a separate function so each can be tested and retried on its own:

  search_pubmed(query) -> PMIDs (PubMed, because only it has real MeSH indexing
                          and a working review[pt]; db=pmc searches full text)
  to_pmcids(pmids)     -> the subset of those held in PMC
  fetch_article(id)    -> raw JATS XML for one article
  parse_article(xml)   -> an Article, or None if it fails the licence filter

Licence policy: ND variants are always rejected, since NoDerivatives conflicts with
chunking and reassembling text. NC variants are accepted (Claricyte is
non-commercial). PMC's custom "available for text mining" terms are accepted but
recorded under their own value, because they grant mining outright while leaving
redistribution to fair use, and keeping them distinct means one predicate can drop
that bucket later.

Uses only the standard library (urllib, ElementTree) rather than requests/lxml, so
the corpus build adds no runtime dependency to the hosted demo.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# NCBI allows 3 requests/second unauthenticated. Sleeping between calls is politer
# and more reliable than getting throttled and retrying.
REQUEST_INTERVAL_SECONDS = 0.34

# Licences we accept. ND variants are excluded throughout: NoDerivatives conflicts
# with chunking and reassembling text, which is the one thing this pipeline must do.
# NC variants are accepted because Claricyte is non-commercial; the corpus is
# separately licensed from the MIT code and the README states its terms.
ALLOWED_LICENCES = (
    "CC0",
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
)

# PMC's custom terms: "available for text mining ... may also be used consistent
# with the principles of fair use". Mining is granted outright; redistribution is
# not, and rests on fair use (short attributed excerpts, linked, non-commercial).
# Recorded as its own value rather than flattened into a CC licence so the corpus
# carries the distinction and one filter can drop this bucket later.
TEXT_MINING_LICENCE = "PMC text-mining"

# JATS section types that are never worth retrieving: boilerplate, or text about the
# study rather than about the biology. Reference lists in particular would flood the
# index with author names and journal titles.
SKIP_SECTION_TYPES = {
    "COI-statement",
    "conflict",
    "data-availability",
    "ethics",
    "funding",
    "supplementary-material",
}
SKIP_HEADING_WORDS = (
    "acknowledg",
    "author contribution",
    "competing interest",
    "conflict of interest",
    "data availability",
    "funding",
    "references",
    "supplementary",
)

# Used when a section carries no heading of its own. Chroma metadata cannot hold
# None, and a missing key would force every downstream consumer to handle two shapes.
NO_SECTION = "untitled"


@dataclass(frozen=True)
class Article:
    """One parsed, license-cleared article, ready to be chunked.

    ``sections`` is ordered as the paper is, each entry a (heading, text) pair.
    Chunking happens per section and never across a boundary, so the heading stays
    attached to the passages it covers and can be matched by the retrieval eval.
    """

    pmcid: str
    title: str
    url: str
    license: str
    sections: tuple[tuple[str, str], ...]


def _get(url: str, retries: int = 3) -> bytes:
    """GET a URL, retrying on transient failure with a widening backoff.

    Raises the final exception rather than returning None: a fetch that fails after
    three attempts is a real problem the caller needs to see, not a silent gap in
    the corpus.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retried, then re-raised below
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(
        f"failed to fetch {url} after {retries} attempts"
    ) from last_error


def search_pmc(
    query: str, max_results: int = 100, sort: str = "relevance"
) -> list[str]:
    """Return PMCIDs matching `query`, restricted to the open-access subset.

    The ``open access[filter]`` clause is appended to the query so non-OA articles
    never enter the pipeline; efetch would return only their abstracts anyway.

    Sorted by relevance rather than the default (date), which otherwise returns
    whatever was published most recently that happens to mention the terms. Note
    that db=pmc searches FULL TEXT, so an unrestricted query matches any paper that
    mentions a word once: field tags like ``neutrophil[TI]`` do the real work here,
    and belong in the query itself.
    """
    params = urllib.parse.urlencode(
        {
            "db": "pmc",
            "term": f"({query}) AND open access[filter]",
            "retmax": max_results,
            "retmode": "xml",
            "sort": sort,
        }
    )
    root = ET.fromstring(_get(f"{EUTILS}/esearch.fcgi?{params}"))
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return [f"PMC{element.text}" for element in root.iter("Id") if element.text]


def fetch_article(pmcid: str) -> bytes:
    """Return the raw JATS XML for one article."""
    params = urllib.parse.urlencode(
        {"db": "pmc", "id": pmcid.removeprefix("PMC"), "retmode": "xml"}
    )
    xml = _get(f"{EUTILS}/efetch.fcgi?{params}")
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return xml


def _licence_of(root: ET.Element) -> str | None:
    """Extract a normalised licence string, or None if it is not machine-readable.

    Publishers declare the licence inconsistently, so all three known carriers are
    checked: a ``license-type`` attribute, an ``xlink:href``, and the text of an
    ``ali:license_ref`` child (the NISO ALI form, which is what PMC actually emits
    for most modern articles). The Creative Commons URL is the reliable signal, so
    it is matched first and the attribute form is only a fallback.

    Returns None for any licence outside ALLOWED_LICENCES, which includes every
    NC and ND variant and every article with no machine-readable licence at all.
    """
    # Creative Commons URL path -> our normalised name. Anything absent from this
    # map (by-nc, by-nd, by-nc-sa, ...) is not redistributable on our terms.
    cc_codes = {
        "by": "CC BY",
        "by-sa": "CC BY-SA",
        "by-nc": "CC BY-NC",
        "by-nc-sa": "CC BY-NC-SA",
        "zero": "CC0",
    }

    for licence in root.iter("license"):
        # Everything the element carries, attributes and descendant text alike.
        declared = " ".join(
            [
                licence.get("license-type", ""),
                licence.get("{http://www.w3.org/1999/xlink}href", ""),
                *(node.text or "" for node in licence.iter()),
                *(value for node in licence.iter() for value in node.attrib.values()),
            ]
        ).lower()

        match = re.search(
            r"creativecommons\.org/(?:licenses|publicdomain)/([a-z-]+)", declared
        )
        if match:
            return cc_codes.get(match.group(1))

        # PMC's custom text-mining terms carry no CC URL, so match the wording.
        if "available for text mining" in declared:
            return TEXT_MINING_LICENCE

        # No URL: fall back to the bare attribute form ("open-access", "CC BY").
        for allowed in sorted(ALLOWED_LICENCES, key=len, reverse=True):
            if allowed.lower() in declared:
                return allowed
    return None


# Subtrees whose content is noise in a retrieval passage: citation markers, and
# tables and figures whose text makes no sense stripped of its layout.
SKIP_SUBTREES = ("xref", "table-wrap", "fig")


def _text_of(element: ET.Element) -> str:
    """Flatten an element's text, dropping citation markers, tables and figures.

    Recurses rather than using iter() so a skipped element skips its whole subtree
    but keeps its tail. The tail is the text that FOLLOWS the element and belongs to
    the parent's flow, so dropping it with the element silently truncates every
    sentence that contains an inline citation.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.tag not in SKIP_SUBTREES:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return " ".join(" ".join(parts).split())


def _is_skippable(section_type: str, heading: str) -> bool:
    """True for boilerplate sections that should never enter the corpus."""
    if section_type in SKIP_SECTION_TYPES:
        return True
    lowered = heading.lower()
    return any(word in lowered for word in SKIP_HEADING_WORDS)


def parse_article(xml: bytes) -> Article | None:
    """Parse JATS XML into an Article, or return None if it should be dropped.

    Returns None (rather than raising) for the two expected, non-exceptional
    rejections: a licence outside ALLOWED_LICENCES, and an article with no body.
    The second is a real trap: efetch answers for non-OA articles with metadata and
    an abstract instead of failing, so a missing body has to be checked explicitly
    or the corpus quietly fills with abstract-only stubs.
    """
    root = ET.fromstring(xml)

    licence = _licence_of(root)
    if licence is None:
        return None

    body = root.find(".//body")
    if body is None or not len(body):
        return None

    # PMC tags this as "pmcid" and already includes the PMC prefix in the value,
    # but "pmcaid" (bare digits) appears in older records, so both are accepted and
    # the prefix is normalised rather than assumed.
    pmcid_element = root.find(".//article-id[@pub-id-type='pmcid']")
    if pmcid_element is None:
        pmcid_element = root.find(".//article-id[@pub-id-type='pmcaid']")
    if pmcid_element is None or not pmcid_element.text:
        return None
    pmcid = f"PMC{pmcid_element.text.strip().removeprefix('PMC')}"

    title_element = root.find(".//title-group/article-title")
    title = _text_of(title_element) if title_element is not None else pmcid

    sections: list[tuple[str, str]] = []
    for section in body.iter("sec"):
        heading_element = section.find("title")
        heading = _text_of(heading_element) if heading_element is not None else ""
        if _is_skippable(section.get("sec-type", ""), heading):
            continue
        # Only this section's own paragraphs: iter() would pull in the text of every
        # nested subsection too, duplicating it once per level of nesting.
        text = " ".join(_text_of(p) for p in section.findall("p"))
        if text.strip():
            sections.append((heading or NO_SECTION, text))

    if not sections:
        return None

    return Article(
        pmcid=pmcid,
        title=title,
        url=ARTICLE_URL.format(pmcid=pmcid),
        license=licence,
        sections=tuple(sections),
    )


def search_pubmed(query: str, max_results: int = 15) -> list[str]:
    """Return PMIDs matching `query`, restricted to articles held in PMC.

    PubMed rather than PMC because only PubMed has real MeSH indexing and a
    working ``review[pt]`` filter; db=pmc searches full text, so any paper that
    mentions a word once matches. ``pubmed pmc[sb]`` is the subset filter that
    actually works: "open access"[filter] is not a valid tag and silently returns
    zero results for every query.
    """
    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": f"({query}) AND pubmed pmc[sb]",
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
    )
    payload = json.loads(_get(f"{EUTILS}/esearch.fcgi?{params}"))
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return payload.get("esearchresult", {}).get("idlist", [])


def to_pmcids(pmids: list[str]) -> list[str]:
    """Convert PMIDs to PMCIDs, dropping any with no PMC record."""
    if not pmids:
        return []
    params = urllib.parse.urlencode({"ids": ",".join(pmids), "format": "json"})
    payload = json.loads(_get(f"{IDCONV}?{params}"))
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return [r["pmcid"] for r in payload.get("records", []) if r.get("pmcid")]
