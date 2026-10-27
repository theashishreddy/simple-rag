"""Vector store implementation with cosine similarity search and JSON persistence."""

from pathlib import Path
from typing import List, Optional, Union
import json
import numpy as np

from simple_rag.models import Chunk, SearchResult


class SimpleVectorStore:
    """Lightweight in-memory vector database with fast numpy cosine similarity search and persistence."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # Shape: (N, D)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks (with embeddings) to the vector store."""
        new_valid_chunks = [c for c in chunks if c.embedding is not None]
        if not new_valid_chunks:
            return

        self.chunks.extend(new_valid_chunks)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Build normalized numpy matrix for vectorized cosine similarity."""
        if not self.chunks:
            self._matrix = None
            return

        vectors = np.array([c.embedding for c in self.chunks], dtype=np.float32)
        # Normalize each row to unit length so dot product is exact cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = vectors / norms

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Perform cosine similarity search against stored embeddings."""
        if self._matrix is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm
        else:
            return []

        # Vectorized dot product (cosine similarity since both are unit normalized)
        similarities = np.dot(self._matrix, q_vec)

        # Get indices sorted by similarity descending
        ranked_indices = np.argsort(similarities)[::-1]

        results: List[SearchResult] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_similarity:
                break
            results.append(SearchResult(chunk=self.chunks[idx], score=score))
            if len(results) >= top_k:
                break

        return results

    def save(self, file_path: Union[str, Path]) -> None:
        """Save vector store to a JSON file."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }
            for chunk in self.chunks
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path: Union[str, Path]) -> None:
        """Load vector store from a JSON file."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Vector store file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [
            Chunk(
                id=item["id"],
                document_id=item["document_id"],
                content=item["content"],
                metadata=item.get("metadata", {}),
                embedding=item.get("embedding"),
            )
            for item in data
        ]
        self._rebuild_matrix()

    def clear(self) -> None:
        """Empty the vector store."""
        self.chunks.clear()
        self._matrix = None

    @property
    def count(self) -> int:
        """Total chunks stored."""
        return len(self.chunks)
