"""Tests for vector store similarity search and persistence."""

from pathlib import Path
from simple_rag.models import Chunk
from simple_rag.vector_store import SimpleVectorStore


def test_vector_store_similarity_search():
    store = SimpleVectorStore()

    # Three 3-dimensional orthogonal/directional vectors
    c1 = Chunk(id="1", document_id="d1", content="Chunk 1", embedding=[1.0, 0.0, 0.0])
    c2 = Chunk(id="2", document_id="d1", content="Chunk 2", embedding=[0.0, 1.0, 0.0])
    c3 = Chunk(id="3", document_id="d2", content="Chunk 3", embedding=[0.7, 0.7, 0.0])

    store.add_chunks([c1, c2, c3])
    assert store.count == 3

    # Query matching vector 1 exactly
    results = store.similarity_search(query_embedding=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "1"
    assert results[0].score >= 0.99
    assert results[1].chunk.id == "3"


def test_vector_store_persistence(tmp_path: Path):
    store = SimpleVectorStore()
    c1 = Chunk(
        id="101",
        document_id="doc-a",
        content="Persisted content",
        metadata={"filename": "doc.md"},
        embedding=[0.5, 0.5, 0.0],
    )
    store.add_chunks([c1])

    save_path = tmp_path / "test_store.json"
    store.save(save_path)
    assert save_path.exists()

    new_store = SimpleVectorStore()
    new_store.load(save_path)

    assert new_store.count == 1
    assert new_store.chunks[0].id == "101"
    assert new_store.chunks[0].content == "Persisted content"
    assert new_store.chunks[0].metadata["filename"] == "doc.md"

    # Search should still work identically
    res = new_store.similarity_search([0.5, 0.5, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0].chunk.id == "101"
