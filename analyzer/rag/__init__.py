"""RAG (Retrieval-Augmented Generation) module for Code-Analyzer.

This module provides semantic search and natural language Q&A capabilities
over analyzed codebases using vector embeddings and LLM integration.
"""

from analyzer.rag.config import RAGConfig
from analyzer.rag.chunker import CodeChunker, CodeChunk
from analyzer.rag.embeddings import (
    EmbeddingProvider,
    get_embedding_provider,
)
from analyzer.rag.vector_store import (
    VectorStore,
    SearchResult,
    get_vector_store,
)
from analyzer.rag.retriever import Retriever, RetrievalResult
from analyzer.rag.llm_provider import LLMProvider, get_llm_provider
from analyzer.rag.pipeline import RAGPipeline, RAGResponse

__all__ = [
    # Config
    "RAGConfig",
    # Chunking
    "CodeChunker",
    "CodeChunk",
    # Embeddings
    "EmbeddingProvider",
    "get_embedding_provider",
    # Vector Store
    "VectorStore",
    "SearchResult",
    "get_vector_store",
    # Retrieval
    "Retriever",
    "RetrievalResult",
    # LLM
    "LLMProvider",
    "get_llm_provider",
    # Pipeline
    "RAGPipeline",
    "RAGResponse",
]
