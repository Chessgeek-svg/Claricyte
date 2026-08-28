"""Vector store: embedding and retrieval over the chunk corpus.

Everything is lazy. Loading the embedder and building the collection costs
~260MB, and the hosted demo has roughly 1GB total with the CBM already using
650MB. The quiz path never retrieves, so nothing here loads until someone asks a
question.

What ships is the corpus plus a 1.5MB npz of its vectors.
Embedding 1050 chunks takes 102s, so it cannot happen at startup, but rebuilding
the collection from precomputed vectors takes 0.9s. A persisted Chroma directory
would be 19MB of binary that churns entirely whenever chunk boundaries move.

Embeddings are computed here rather than through a Chroma embedding_function, so
indexing and querying use the same model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from claricyte.rag.corpus import Chunk, read_jsonl

# 384-dim, ~130MB on first download. Vectors from different models are not
# comparable, so changing this invalidates the npz.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "claricyte_corpus"
CORPUS_PATH = "rag_data/corpus.jsonl"
EMBEDDINGS_PATH = "rag_data/embeddings.npz"

# BGE was trained with an instruction prefix on queries but not on documents.
# Omitting it costs a couple of points of retrieval accuracy.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder():
    """The sentence-transformer, loaded once per process on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts, L2-normalised so cosine distance is a dot product."""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return _embedder().encode(texts, normalize_embeddings=True).astype(np.float32)


def build_embeddings(
    corpus_path: str | Path = CORPUS_PATH, out_path: str | Path = EMBEDDINGS_PATH
) -> int:
    """Embed a corpus and save its vectors. Offline only: ~100s for 1000 chunks.

    Ids are saved alongside so vectors stay matched to their chunks regardless of
    corpus ordering, and a stale npz fails loudly rather than silently misaligning.
    """
    chunks = read_jsonl(corpus_path)
    vectors = embed([chunk.text for chunk in chunks])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, vectors=vectors, ids=np.array([chunk.id for chunk in chunks])
    )
    return len(chunks)


@lru_cache(maxsize=1)
def _collection(corpus_path: str = CORPUS_PATH, embeddings_path: str = EMBEDDINGS_PATH):
    """Build the in-memory collection from the corpus and its vectors."""
    import chromadb

    chunks = read_jsonl(corpus_path)
    saved = np.load(embeddings_path)
    position = {chunk_id: i for i, chunk_id in enumerate(saved["ids"])}

    missing = [c.id for c in chunks if c.id not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} chunks have no saved vector (e.g. {missing[0]}). "
            "Rerun scripts/build_index.py after changing the corpus."
        )

    collection = chromadb.EphemeralClient().create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    triples = [chunk.chroma_keys() for chunk in chunks]
    ids, documents, metadatas = zip(*triples)
    vectors = saved["vectors"]
    for start in range(0, len(ids), 256):
        stop = start + 256
        collection.add(
            ids=list(ids[start:stop]),
            documents=list(documents[start:stop]),
            metadatas=list(metadatas[start:stop]),
            embeddings=[vectors[position[i]].tolist() for i in ids[start:stop]],
        )
    return collection


def search(
    text: str, where: dict | None = None, k: int = 5
) -> list[tuple[Chunk, float]]:
    """Retrieve the k nearest chunks to `text`, optionally filtered by metadata.

    Returns (chunk, similarity) pairs, most similar first, where similarity is
    1 - cosine distance.
    """
    result = _collection().query(
        query_embeddings=embed([text], is_query=True).tolist(),
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        (_to_chunk(chunk_id, document, metadata), 1.0 - distance)
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def _to_chunk(chunk_id: str, document: str, metadata: dict) -> Chunk:
    """Rebuild a Chunk from what Chroma stored.

    chunk_index comes back off the id rather than being stored twice, since Chunk
    derives the id from it and the two must not be able to disagree.
    """
    return Chunk(
        text=document,
        pmcid=metadata["pmcid"],
        section=metadata["section"],
        url=metadata["url"],
        license=metadata["license"],
        cell_classes=tuple(metadata["cell_classes"]),
        title=metadata["title"],
        chunk_index=int(chunk_id.rsplit(":", 1)[1]),
    )
