"""Data models for Simple RAG."""

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class Document:
    """Represents an ingested raw document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Chunk:
    """Represents a chunk of a document after splitting."""

    content: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    """Result of a similarity search in the vector store."""

    chunk: Chunk
    score: float  # Cosine similarity score (0.0 to 1.0)


@dataclass
class RAGResponse:
    """Complete response returned by the RAG pipeline."""

    query: str
    answer: str
    sources: list[SearchResult]
    retrieval_time_ms: float
    generation_time_ms: float
    model_used: str
