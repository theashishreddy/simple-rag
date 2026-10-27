from dotenv import load_dotenv

load_dotenv()

from simple_rag.models import Document, Chunk, SearchResult, RAGResponse
from simple_rag.document_loader import DocumentLoader
from simple_rag.text_splitter import RecursiveCharacterTextSplitter
from simple_rag.embeddings import (
    BaseEmbeddingModel,
    OpenAIEmbeddingModel,
    LocalTFIDFEmbeddingModel,
    get_embedding_model,
)
from simple_rag.vector_store import SimpleVectorStore
from simple_rag.llm import BaseLLM, OpenAILLM, LocalMockLLM, get_llm
from simple_rag.pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "Document",
    "Chunk",
    "SearchResult",
    "RAGResponse",
    "DocumentLoader",
    "RecursiveCharacterTextSplitter",
    "SimpleVectorStore",
    "BaseEmbeddingModel",
    "OpenAIEmbeddingModel",
    "LocalTFIDFEmbeddingModel",
    "get_embedding_model",
    "BaseLLM",
    "OpenAILLM",
    "LocalMockLLM",
    "get_llm",
    "RAGPipeline",
]
