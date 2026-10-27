"""Unified RAG Pipeline coordinating ingestion, retrieval, augmentation, and generation."""

from pathlib import Path
import time
from typing import Iterator, List, Optional, Tuple, Union

from simple_rag.models import Document, Chunk, SearchResult, RAGResponse
from simple_rag.document_loader import DocumentLoader
from simple_rag.text_splitter import RecursiveCharacterTextSplitter
from simple_rag.embeddings import BaseEmbeddingModel, get_embedding_model
from simple_rag.vector_store import SimpleVectorStore
from simple_rag.llm import BaseLLM, get_llm


DEFAULT_SYSTEM_PROMPT = """You are an accurate, helpful assistant equipped with a retrieval system.
Answer the user's question using ONLY the provided Context sections below.
- Ground your answer firmly in the provided context.
- If the answer cannot be determined from the context, clearly state: "Based on the provided documentation, I do not have enough information to answer that question."
- Explicitly cite the source filename or section when referencing specific facts (e.g. "[Source: filename.md]").
- Keep your tone objective, professional, and concise.
"""


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        llm: Optional[BaseLLM] = None,
        vector_store: Optional[SimpleVectorStore] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.embedding_model = embedding_model or get_embedding_model()
        self.llm = llm or get_llm()
        self.vector_store = vector_store or SimpleVectorStore()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.system_prompt = system_prompt

    def ingest_documents(self, documents: List[Document]) -> int:
        """Split documents, compute embeddings, and insert into vector store."""
        if not documents:
            return 0

        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.vector_store.add_chunks(chunks)
        return len(chunks)

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
    ) -> dict:
        """Load and index all supported documents in a directory."""
        docs = DocumentLoader.load_directory(dir_path, recursive=recursive)
        num_chunks = self.ingest_documents(docs)
        return {
            "documents_loaded": len(docs),
            "chunks_created": num_chunks,
            "total_chunks_in_store": self.vector_store.count,
        }

    def ingest_file(self, file_path: Union[str, Path]) -> dict:
        """Load and index a single document file."""
        doc = DocumentLoader.load_file(file_path)
        num_chunks = self.ingest_documents([doc])
        return {
            "documents_loaded": 1,
            "chunks_created": num_chunks,
            "total_chunks_in_store": self.vector_store.count,
        }

    def ingest_text(self, text: str, source_name: str = "inline_snippet.txt") -> dict:
        """Index raw text directly."""
        doc = Document(
            content=text,
            metadata={"source": source_name, "filename": source_name, "char_count": len(text)},
        )
        num_chunks = self.ingest_documents([doc])
        return {
            "documents_loaded": 1,
            "chunks_created": num_chunks,
            "total_chunks_in_store": self.vector_store.count,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Retrieve the most relevant context chunks for a query."""
        if self.vector_store.count == 0:
            return []

        query_emb = self.embedding_model.embed_query(query)
        return self.vector_store.similarity_search(
            query_emb, top_k=top_k, min_similarity=min_similarity
        )

    def format_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """Format retrieved chunks into a structured augmented prompt."""
        context_parts = []
        for i, res in enumerate(search_results, 1):
            source_file = res.chunk.metadata.get("filename", "Unknown")
            similarity_pct = f"{res.score * 100:.1f}%"
            context_parts.append(
                f"[Document {i} | Source: {source_file} | Relevance: {similarity_pct}]\n{res.chunk.content}"
            )

        context_block = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No relevant documents found in knowledge base."
        )

        return (
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query}\n\n"
            f"Helpful Answer:"
        )

    def query(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> RAGResponse:
        """Full RAG query execution."""
        t0 = time.perf_counter()
        results = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        prompt = self.format_prompt(query, results)

        t1 = time.perf_counter()
        answer = self.llm.generate(prompt=prompt, system_prompt=self.system_prompt)
        generation_ms = (time.perf_counter() - t1) * 1000.0

        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            model_used=self.llm.model_name,
        )

    def stream_query(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> Tuple[Iterator[str], List[SearchResult]]:
        """Streaming version of RAG query returning an iterator for tokens and retrieved sources."""
        results = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)
        prompt = self.format_prompt(query, results)
        stream_iter = self.llm.stream(prompt=prompt, system_prompt=self.system_prompt)
        return stream_iter, results

    def save_index(self, path: Union[str, Path]) -> None:
        """Persist vector store index to disk."""
        self.vector_store.save(path)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load vector store index from disk."""
        self.vector_store.load(path)
