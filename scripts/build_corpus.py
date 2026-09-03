"""Build the retrieval corpus from the PubMed Central open-access subset.

    python scripts/build_corpus.py --out rag_data/corpus.jsonl

Pipeline per query: PubMed search (real MeSH indexing and a working review[pt],
which db=pmc lacks) -> convert PMIDs to PMCIDs -> fetch JATS -> parse and apply
the licence filter -> chunk -> tag by cell class -> write jsonl.

Articles are dropped, loudly and with a counted reason, when the licence is not
redistributable or the body is missing. Chunks are dropped when they tag to no
class, since nothing could ever retrieve them through the class filter.

Search terms below target clinical correlation rather than morphology alone: what
an abnormal count means, what it is confused with, and what artefacts mimic it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from claricyte.rag.books import BOOKS, load_book
from claricyte.rag.corpus import chunk_article, tag_classes, write_jsonl
from claricyte.rag.pmc import fetch_article, parse_article, search_pubmed, to_pmcids

# Appended to every search. The standard PubMed idiom for excluding animal-only
# studies: it drops veterinary and pure mouse-model papers while keeping articles
# too recent to have been MeSH indexed, which a bare humans[mh] would lose.
# Without it the corpus took in rabbit differential counts and a small-animal
# bacteraemia paper, which retrieval then cited under Basophil.
HUMAN_ONLY = " NOT (animals[mh] NOT humans[mh])"

SEARCHES: dict[str, str] = {
    "neutrophilia": (
        'neutrophilia AND (causes OR "differential diagnosis" OR approach)'
        " AND review[pt]"
    ),
    "neutropenia": (
        'neutropenia AND (causes OR "differential diagnosis" OR approach'
        " OR evaluation) AND review[pt]"
    ),
    "lymphocytosis": (
        'lymphocytosis AND (causes OR "differential diagnosis" OR approach)'
        " AND review[pt]"
    ),
    "lymphopenia": (
        'lymphopenia AND (causes OR "differential diagnosis"'
        ' OR "clinical significance") AND review[pt]'
    ),
    "monocytosis": (
        'monocytosis AND (causes OR "differential diagnosis"'
        ' OR "clinical significance") AND review[pt]'
    ),
    "eosinophilia": (
        'eosinophilia AND (causes OR "differential diagnosis" OR approach'
        " OR workup) AND review[pt]"
    ),
    "basophilia": (
        'basophilia AND (causes OR "differential diagnosis" OR "clinical significance")'
    ),
    "leukocytosis": (
        'leukocytosis AND (causes OR "differential diagnosis" OR approach)'
        " AND review[pt]"
    ),
    "reactive_lymphocytes": (
        '("reactive lymphocytes" OR "atypical lymphocytes")'
        ' AND (morphology OR "differential diagnosis")'
    ),
    # The lymphocyte panel came out almost entirely COVID, because the recent
    # open-access literature on lymphopenia is COVID. This asks for the causes
    # directly rather than hoping a general query surfaces them.
    "reactive_lymphocytosis": (
        '("reactive lymphocytosis" OR "benign lymphocytosis"'
        ' OR "absolute lymphocytosis") AND (causes OR evaluation OR approach'
        ' OR "differential diagnosis")'
    ),
    "left_shift": (
        '("left shift" OR "band neutrophil" OR "immature granulocytes")'
        ' AND ("clinical significance" OR "peripheral blood") AND review[pt]'
    ),
    "toxic_change": (
        '("toxic granulation" OR "toxic change" OR "Dohle bodies")'
        ' AND (morphology OR "peripheral blood")'
    ),
    "smear_artifact": (
        '("blood smear" OR "peripheral smear")'
        " AND (artifact OR preanalytical OR EDTA)"
        ' AND (leukocyte OR "white blood cell")'
    ),
    "morphology_review": (
        '("blood film" OR "peripheral blood smear")'
        ' AND (morphology OR "morphologic review") AND review[pt]'
    ),
}

# Hand-picked articles that relevance ranking misses. Always fetched, so a known
# good source cannot be lost to a reshuffle when a query or --per-query changes.
# This is the mechanism for acting on any curated recommendation.
EXTRA_PMCIDS: tuple[str, ...] = (
    "PMC11130981",  # The challenge of diagnosing and classifying eosinophilia
    "PMC10814743",  # Hematological Neoplasms with Eosinophilia
    "PMC11270355",  # Transient Stress Lymphocytosis: case report and review
    "PMC10148979",  # French guidelines for the etiological workup of eosinophilia
    # Lymphocyte sources, picked by hand after ten queries scored on descriptive
    # content rather than hit count. Open-access reviews are written for people
    # who already know what a reactive lymphocyte looks like, so relevance
    # ranking surfaces molecular immunology and these have to be named.
    "PMC10527541",  # Atypical chronic lymphocytic leukaemia, the current status
    "PMC8255663",  # Lymphocytosis with smudge cells is not equivalent to CLL
    "PMC8418501",  # High fluorescent lymphocytes and smudge cells
    "PMC5336551",  # Chronic lymphocytic leukaemia
    "PMC4177785",  # New insights into monoclonal B-cell lymphocytosis
    "PMC12436449",  # CMV infection-induced lymphocytosis
    "PMC12734239",  # The many faces of primary EBV infection
    "PMC10140754",  # Mycosis fungoides and Sezary syndrome
    # Cell physiology, which the corpus had almost none of. Asked why a
    # neutrophil nucleus is segmented, the model gave the textbook answer about
    # deformability and migration and cited two chunks that say nothing of the
    # kind. The searches above look for clinical correlation, so nothing that
    # explains why a cell is shaped the way it is was ever going to be found.
    "PMC6288403",  # The neutrophil nucleus: an important influence on migration
    "PMC6250837",  # Nuclear deformation during neutrophil migration
)


# Articles a search keeps returning that should not be in a corpus about human
# blood films. The MeSH filter above misses these: it excludes articles indexed
# as animal-only, and an article too recent to be indexed at all is not. Nothing
# subtle is being judged here, only species.
REJECTED_PMCIDS: frozenset[str] = frozenset(
    {
        "PMC12173895",  # Bacteraemia on peripheral blood smear in small animals
        "PMC8225691",  # Differential white blood cell counts in rabbits
        "PMC6430879",  # Comparative pathophysiology of protein-losing enteropathy
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="rag_data/corpus.jsonl")
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()

    # Dedupe across topics: the same review answers several of these queries.
    pmcids: list[str] = [p for p in EXTRA_PMCIDS if p not in REJECTED_PMCIDS]
    print(f"{'curated':22} {len(pmcids):3} pinned")
    for topic, query in SEARCHES.items():
        found = to_pmcids(search_pubmed(query + HUMAN_ONLY, args.per_query))
        fresh = [p for p in found if p not in pmcids and p not in REJECTED_PMCIDS]
        pmcids.extend(fresh)
        print(f"{topic:22} {len(found):3} hits, {len(fresh):3} new")

    print(f"\n{len(pmcids)} unique articles to fetch\n")

    chunks = []
    dropped: Counter[str] = Counter()
    for i, pmcid in enumerate(pmcids, 1):
        try:
            article = parse_article(fetch_article(pmcid))
        except Exception as error:
            dropped["fetch failed"] += 1
            print(f"  [{i}/{len(pmcids)}] {pmcid} fetch failed: {error}")
            continue
        if article is None:
            dropped["licence or no body"] += 1
            continue
        # Tagged per chunk, not per article: a paper's methods section is not about
        # the same cell as its discussion.
        for chunk in chunk_article(article, cell_classes=(), max_words=args.max_words):
            labels = tag_classes(chunk.text)
            if not labels:
                dropped["chunk untagged"] += 1
                continue
            chunks.append(replace(chunk, cell_classes=labels))

    # Open-licensed textbooks, a second front door onto the same Article shape.
    # These carry the descriptive morphology the journal corpus lacks: the
    # band-versus-seg criteria, toxic change, Dohle bodies.
    for book in BOOKS:
        article = load_book(book)
        before = len(chunks)
        for chunk in chunk_article(article, cell_classes=(), max_words=args.max_words):
            labels = tag_classes(chunk.text)
            if not labels:
                dropped["chunk untagged"] += 1
                continue
            chunks.append(replace(chunk, cell_classes=labels))
        print(
            f"{book.source_id:22} {len(article.sections):3} sections, "
            f"{len(chunks) - before:3} chunks kept"
        )

    write_jsonl(chunks, args.out)

    kept = {c.source_id for c in chunks}
    print(f"\nkept {len(kept)} articles, {len(chunks)} chunks -> {args.out}")
    for reason, count in dropped.most_common():
        print(f"  dropped ({reason}): {count}")
    tags = Counter(label for c in chunks for label in c.cell_classes)
    print("\nchunks per tag:")
    for label, count in tags.most_common():
        print(f"  {label:22} {count}")
    licences = Counter(c.license for c in chunks)
    print(f"\nlicences: {dict(licences)}")


if __name__ == "__main__":
    main()
