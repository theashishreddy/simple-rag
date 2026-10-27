"""Embedding models for Simple RAG: OpenAI and offline TF-IDF/hashed vectorizer."""

from abc import ABC, abstractmethod
import math
import os
import re
from typing import List, Optional
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI API based embedding generator (text-embedding-3-small default)."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimension = 1536 if "small" in model_name or "ada" in model_name else 3072

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts in batches."""
        if not texts:
            return []

        cleaned_texts = [t.replace("\n", " ").strip() or " " for t in texts]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i : i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        cleaned = text.replace("\n", " ").strip() or " "
        response = self.client.embeddings.create(input=[cleaned], model=self.model_name)
        return response.data[0].embedding


class LocalTFIDFEmbeddingModel(BaseEmbeddingModel):
    """Offline, zero-API-cost embedding model using feature hashing & TF-IDF token weighting.

    Generates normalized dense vectors suitable for cosine similarity.
    """

    def __init__(self, dimension: int = 256):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w{2,}\b", text.lower())

    def _hash_token(self, token: str) -> int:
        return abs(hash(token)) % self._dim

    def _vectorize(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        vec = np.zeros(self._dim, dtype=np.float32)
        token_counts: dict[str, int] = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        total_tokens = len(tokens)
        for token, count in token_counts.items():
            idx = self._hash_token(token)
            tf = count / total_tokens
            # Sub-linear term frequency boost
            weight = 1.0 + math.log(1.0 + tf)
            vec[idx] += weight

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)


def is_valid_openai_key(key: Optional[str]) -> bool:
    if not key:
        return False
    key = key.strip()
    return (key.startswith("sk-") or key.startswith("sess-")) and len(key) >= 20


def get_embedding_model(provider: str = "auto") -> BaseEmbeddingModel:
    """Factory to retrieve embedding model based on provider preference and environment."""
    provider = provider.lower()
    raw_key = os.environ.get("OPENAI_API_KEY", "")
    has_valid_key = is_valid_openai_key(raw_key)

    if provider == "openai":
        return OpenAIEmbeddingModel()
    elif provider == "local":
        return LocalTFIDFEmbeddingModel()
    elif provider == "auto":
        if has_valid_key:
            return OpenAIEmbeddingModel()
        else:
            return LocalTFIDFEmbeddingModel()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Options: 'auto', 'openai', 'local'")
