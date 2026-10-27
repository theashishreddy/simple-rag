"""Tests for document chunking and text splitting."""

import pytest
from simple_rag.models import Document
from simple_rag.text_splitter import RecursiveCharacterTextSplitter


def test_splitter_basic():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "Sentence one is here. " * 10
    doc = Document(content=text, metadata={"filename": "test.txt"})

    chunks = splitter.split_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 120  # Allows small boundary tolerance
        assert chunk.document_id == doc.id
        assert chunk.metadata["filename"] == "test.txt"
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["total_chunks"] == len(chunks)


def test_splitter_short_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    doc = Document(content="Short text.")
    chunks = splitter.split_document(doc)

    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_splitter_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
