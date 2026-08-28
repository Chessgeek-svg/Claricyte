"""Embed the corpus and save its vectors.

    python scripts/build_index.py

Offline: ~100s for 1000 chunks on CPU. Rerun whenever corpus.jsonl changes, since
chunk ids move with chunk boundaries and a stale npz will fail loudly at load.
The demo rebuilds the Chroma collection from these vectors in under a second.
"""

import argparse

from claricyte.rag.store import (
    CORPUS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    build_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--out", default=EMBEDDINGS_PATH)
    args = parser.parse_args()

    print(f"embedding {args.corpus} with {EMBEDDING_MODEL} ...")
    count = build_embeddings(args.corpus, args.out)
    print(f"embedded {count} chunks -> {args.out}")


if __name__ == "__main__":
    main()
