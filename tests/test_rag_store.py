"""Tests for the vector store.

Split deliberately. The logic that can fail silently is tested unconditionally
and needs no model: reconstructing a Chunk from what Chroma returns, and the
guard against a corpus and an embeddings file that have drifted apart.

The tests that need BAAI/bge-small are skipped unless it is already cached
locally. Downloading a 130MB model in CI would be testing that a third-party
library works, not that our code does, and it would need network.
"""

import numpy as np
import pytest

from claricyte.rag.corpus import Chunk, write_jsonl
from claricyte.rag.query import class_filter
from claricyte.rag.store import EMBEDDINGS_PATH, _collection, _to_chunk, search


def make_chunk(**overrides) -> Chunk:
    base = dict(
        text="Basophil granules are water soluble.",
        pmcid="PMC7563270",
        section="Morphology",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7563270/",
        license="CC BY",
        cell_classes=("Basophil", "General"),
        title="Basophils in health and disease",
        chunk_index=3,
    )
    base.update(overrides)
    return Chunk(**base)


# --- reconstruction, no model needed --------------------------------------


def test_chunk_survives_the_round_trip_through_chroma():
    """chroma_keys flattens a Chunk; _to_chunk must rebuild it exactly, or every
    citation the UI renders is subtly wrong."""
    original = make_chunk()
    chunk_id, document, metadata = original.chroma_keys()
    assert _to_chunk(chunk_id, document, metadata) == original


def test_chunk_index_is_recovered_from_the_id():
    """It is not stored in metadata, because Chunk derives the id from it and the
    two must not be able to disagree."""
    original = make_chunk(chunk_index=42)
    chunk_id, document, metadata = original.chroma_keys()
    assert "chunk_index" not in metadata
    assert _to_chunk(chunk_id, document, metadata).chunk_index == 42


def test_round_trip_survives_a_section_containing_a_colon():
    """The id is split on the LAST colon, so a colon in a heading must not break
    index recovery."""
    original = make_chunk(section="Results: eosinophils", chunk_index=7)
    assert _to_chunk(*original.chroma_keys()).chunk_index == 7


# --- the drift guard, needs chromadb but no model -------------------------


def test_stale_embeddings_raise_rather_than_misalign(tmp_path):
    """Chunk ids move whenever chunk boundaries do. Matching vectors by position
    instead of id would hand every chunk someone else's vector and retrieve
    plausible nonsense with nothing raised anywhere."""
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl([make_chunk(chunk_index=i) for i in range(3)], corpus)
    embeddings = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings,
        vectors=np.zeros((3, 384), dtype=np.float32),
        ids=np.array(["PMC7563270:Morphology:99"] * 3),  # ids from an older build
    )
    with pytest.raises(ValueError, match="no saved vector"):
        _collection(str(corpus), str(embeddings))


# --- retrieval, skipped without the model ---------------------------------


def model_is_cached() -> bool:
    """True when BGE is already downloaded, so tests can run without network."""
    from pathlib import Path

    cache = Path.home() / ".cache/huggingface/hub"
    return (
        Path(EMBEDDINGS_PATH).exists()
        and cache.exists()
        and any(cache.glob("models--BAAI--bge-small-en-v1.5*"))
    )


needs_model = pytest.mark.skipif(
    not model_is_cached(), reason="BGE not cached locally; skipping retrieval tests"
)


@needs_model
def test_search_returns_results_in_descending_similarity():
    hits = search("what conditions cause eosinophilia", k=5)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


@needs_model
def test_search_respects_the_class_filter():
    """Every hit must carry the requested class or GENERAL. A filter that
    silently does nothing would look fine and quietly halve precision."""
    hits = search("elevated count", where=class_filter("Monocyte"), k=20)
    assert hits
    for chunk, _ in hits:
        assert {"Monocyte", "General"} & set(chunk.cell_classes)


@needs_model
def test_search_honours_k():
    assert len(search("eosinophilia", k=3)) == 3
