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

from claricyte.rag.corpus import chunk_article, tag_classes, write_jsonl
from claricyte.rag.pmc import fetch_article, parse_article, search_pubmed, to_pmcids

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="rag_data/corpus.jsonl")
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()

    # Dedupe across topics: the same review answers several of these queries.
    pmcids: list[str] = []
    for topic, query in SEARCHES.items():
        found = to_pmcids(search_pubmed(query, args.per_query))
        fresh = [p for p in found if p not in pmcids]
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

    write_jsonl(chunks, args.out)

    kept = {c.pmcid for c in chunks}
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
