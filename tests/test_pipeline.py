"""End-to-end integration tests for the RAG pipeline."""

from pathlib import Path
from simple_rag.pipeline import RAGPipeline
from simple_rag.embeddings import LocalTFIDFEmbeddingModel
from simple_rag.llm import LocalMockLLM
from simple_rag.vector_store import SimpleVectorStore


def test_pipeline_end_to_end_offline(tmp_path: Path):
    pipeline = RAGPipeline(
        embedding_model=LocalTFIDFEmbeddingModel(),
        llm=LocalMockLLM(),
        vector_store=SimpleVectorStore(),
        chunk_size=200,
        chunk_overlap=40,
    )

    # Ingest snippets
    doc1 = (
        "The Apollo 11 mission was the first spaceflight that landed humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed on July 20, 1969. "
        "Neil Armstrong spent about two and a quarter hours outside the spacecraft."
    )
    doc2 = (
        "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. "
        "It was launched on December 25, 2021 on an Ariane 5 rocket from Kourou, French Guiana."
    )

    pipeline.ingest_text(doc1, source_name="apollo11.txt")
    pipeline.ingest_text(doc2, source_name="jwst.txt")

    assert pipeline.vector_store.count >= 2

    # Query for Apollo 11
    resp = pipeline.query("Who landed on the Moon during Apollo 11?")

    assert resp.sources
    top_source = resp.sources[0]
    assert top_source.chunk.metadata["filename"] == "apollo11.txt"
    assert "Armstrong" in resp.answer or "Apollo 11" in resp.answer

    # Query for JWST
    resp_jwst = pipeline.query("When was the James Webb Space Telescope launched?")
    assert resp_jwst.sources
    assert resp_jwst.sources[0].chunk.metadata["filename"] == "jwst.txt"
