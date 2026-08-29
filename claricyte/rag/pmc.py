"""Search PubMed, fetch from PMC, parse JATS into articles.

All the network and XML code, kept out of the rest of the package. Standard
library only, so building the corpus adds nothing to the hosted demo.

ND licences are rejected: NoDerivatives conflicts with chunking. NC is fine,
Claricyte is non-commercial. PMC's "available for text mining" terms get their
own value, since they grant mining but leave redistribution to fair use.
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
    """A parsed, licence-cleared article.

    sections is (heading, text) pairs in document order. Chunking never crosses a
    heading, so the gold set can match on it.
    """

    source_id: str
    title: str
    url: str
    license: str
    sections: tuple[tuple[str, str], ...]


def _get(url: str, retries: int = 3) -> bytes:
    """GET with backoff. Raises after the last try rather than returning None,
    which would leave a silent hole in the corpus."""
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
    """PMCIDs matching `query`, open-access only, ranked by relevance.

    db=pmc searches full text, so anything mentioning a word once matches. Field
    tags like neutrophil[TI] do the real narrowing and belong in the query.
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
    """Normalised licence name, or None if we cannot redistribute it.

    Publishers declare licences three different ways, so check all of them: the
    license-type attribute, xlink:href, and ali:license_ref text, which is what
    PMC actually emits. The CC URL is the reliable one.
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
    """Element text, minus citation markers, tables and figures.

    Recurses instead of using iter() so a skipped element loses its subtree but
    keeps its tail. The tail is the text after the element, so dropping it
    truncates every sentence containing an inline citation.
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
    """Parse JATS into an Article, or None if it should be dropped.

    None rather than raising, because both rejections are expected: a licence we
    cannot use, or no body. Watch the second one; efetch answers for non-OA
    articles with an abstract instead of failing, so the corpus would fill with
    stubs if we did not check.
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
        source_id=pmcid,
        title=title,
        url=ARTICLE_URL.format(pmcid=pmcid),
        license=licence,
        sections=tuple(sections),
    )


def search_pubmed(query: str, max_results: int = 15) -> list[str]:
    """PMIDs matching `query`, restricted to articles held in PMC.

    PubMed rather than PMC: only it has real MeSH indexing and a working
    review[pt]. Use pubmed pmc[sb] for the subset filter. "open access"[filter]
    is not a valid tag and silently returns nothing at all.
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
