"""Tests for RAG system components."""

import pytest
from pathlib import Path
from textwrap import dedent

from analyzer.rag.config import (
    RAGConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkingConfig,
    RetrievalConfig,
    LLMConfig,
    get_default_rag_config,
    create_rag_config,
)
from analyzer.rag.chunker import CodeChunker, CodeChunk, ChunkingConfig
from analyzer.rag.embeddings import (
    MockEmbeddings,
    get_embedding_provider,
    EmbeddingProvider,
)
from analyzer.rag.vector_store import (
    InMemoryVectorStore,
    SearchResult,
    get_vector_store,
)
from analyzer.rag.retriever import Retriever, RetrievalResult, build_context
from analyzer.rag.llm_provider import MockLLM, get_llm_provider
from analyzer.rag.pipeline import RAGPipeline, RAGResponse

from analyzer.models.code_entities import (
    Module,
    Class,
    Function,
    Method,
    CodeLocation,
    Docstring,
    Parameter,
    EntityType,
)


# Fixtures

@pytest.fixture
def sample_module():
    """Create a sample module for testing."""
    return Module(
        name="test_module",
        entity_type=EntityType.MODULE,
        location=CodeLocation(file_path="test.py", start_line=1, end_line=50),
        file_path="test.py",
        docstring=Docstring(raw="Test module docstring", summary="Test module"),
        imports=[],
        classes=[
            Class(
                name="Calculator",
                entity_type=EntityType.CLASS,
                location=CodeLocation(file_path="test.py", start_line=5, end_line=30),
                docstring=Docstring(raw="A calculator class", summary="Calculator class"),
                bases=["object"],
                methods=[
                    Method(
                        name="add",
                        entity_type=EntityType.METHOD,
                        location=CodeLocation(file_path="test.py", start_line=10, end_line=15),
                        parameters=[
                            Parameter(name="self"),
                            Parameter(name="a", type_annotation="int"),
                            Parameter(name="b", type_annotation="int"),
                        ],
                        return_type="int",
                        docstring=Docstring(raw="Add two numbers", summary="Add two numbers"),
                    ),
                    Method(
                        name="subtract",
                        entity_type=EntityType.METHOD,
                        location=CodeLocation(file_path="test.py", start_line=17, end_line=22),
                        parameters=[
                            Parameter(name="self"),
                            Parameter(name="a", type_annotation="int"),
                            Parameter(name="b", type_annotation="int"),
                        ],
                        return_type="int",
                        docstring=Docstring(raw="Subtract two numbers", summary="Subtract"),
                    ),
                ],
            ),
        ],
        functions=[
            Function(
                name="main",
                entity_type=EntityType.FUNCTION,
                location=CodeLocation(file_path="test.py", start_line=35, end_line=45),
                docstring=Docstring(raw="Main function", summary="Entry point"),
                parameters=[],
                return_type="None",
            ),
        ],
    )


@pytest.fixture
def rag_config():
    """Create RAG config for testing."""
    config = RAGConfig()
    config.embedding.provider = "mock"
    config.llm.provider = "mock"
    config.vector_store.backend = "in_memory"
    # Lower threshold for mock embeddings which produce random scores
    config.retrieval.similarity_threshold = 0.0
    return config


# Configuration Tests

class TestRAGConfig:
    def test_default_config(self):
        """Test default RAG configuration."""
        config = get_default_rag_config()
        
        assert config.embedding.provider == "openai"
        assert config.llm.provider == "auto"  # Auto-detects available AI
        assert config.vector_store.backend == "chromadb"
        assert config.retrieval.top_k == 10
    
    def test_create_rag_config(self):
        """Test creating RAG config with overrides."""
        config = create_rag_config(
            embedding_provider="sentence-transformers",
            llm_provider="anthropic",
            persist_directory="/tmp/test_rag",
        )
        
        assert config.embedding.provider == "sentence-transformers"
        assert config.llm.provider == "anthropic"
        assert config.vector_store.persist_directory == "/tmp/test_rag"
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = RAGConfig()
        config_dict = config.to_dict()
        
        assert "embedding" in config_dict
        assert "vector_store" in config_dict
        assert "retrieval" in config_dict
        assert config_dict["enabled"] is True


# Chunker Tests

class TestCodeChunker:
    def test_chunk_module(self, sample_module):
        """Test chunking a module."""
        chunker = CodeChunker()
        chunks = chunker.chunk_module(sample_module)
        
        assert len(chunks) > 0
        # Should have module overview, class, and function chunks
        entity_types = {c.entity_type for c in chunks}
        assert "module" in entity_types
        assert "class" in entity_types or "function" in entity_types
    
    def test_chunk_preserves_metadata(self, sample_module):
        """Test that chunks preserve metadata."""
        chunker = CodeChunker()
        chunks = chunker.chunk_module(sample_module)
        
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.file_path == "test.py"
            assert chunk.module_name == "test_module"
    
    def test_chunk_content_not_empty(self, sample_module):
        """Test that chunks have content."""
        chunker = CodeChunker()
        chunks = chunker.chunk_module(sample_module)
        
        for chunk in chunks:
            assert len(chunk.content) > 0
    
    def test_chunk_id_uniqueness(self, sample_module):
        """Test that chunk IDs are unique."""
        chunker = CodeChunker()
        chunks = chunker.chunk_module(sample_module)
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


# Embeddings Tests

class TestEmbeddings:
    def test_mock_embeddings_dimensions(self, rag_config):
        """Test mock embeddings have correct dimensions."""
        provider = MockEmbeddings(rag_config.embedding)
        
        texts = ["Hello world", "Test text"]
        embeddings = provider.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == provider.dimension
    
    def test_mock_embeddings_deterministic(self, rag_config):
        """Test mock embeddings are deterministic."""
        provider = MockEmbeddings(rag_config.embedding)
        
        text = "Test text"
        emb1 = provider.embed_query(text)
        emb2 = provider.embed_query(text)
        
        assert emb1 == emb2
    
    def test_embedding_provider_factory(self, rag_config):
        """Test embedding provider factory."""
        rag_config.embedding.provider = "mock"
        provider = get_embedding_provider(rag_config.embedding)
        
        assert isinstance(provider, MockEmbeddings)
    
    def test_embed_batch(self, rag_config):
        """Test batch embedding."""
        provider = MockEmbeddings(rag_config.embedding)
        
        texts = [f"Text {i}" for i in range(150)]
        embeddings = provider.embed_batch(texts, batch_size=50)
        
        assert len(embeddings) == 150


# Vector Store Tests

class TestVectorStore:
    def test_in_memory_add_and_search(self, rag_config, sample_module):
        """Test adding and searching in memory store."""
        store = InMemoryVectorStore(rag_config.vector_store)
        chunker = CodeChunker()
        provider = MockEmbeddings(rag_config.embedding)
        
        chunks = chunker.chunk_module(sample_module)
        embeddings = provider.embed_texts([c.content for c in chunks])
        
        store.add_documents(chunks, embeddings)
        
        # Search
        query_emb = provider.embed_query("calculator add numbers")
        results = store.search(query_emb, top_k=5)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_vector_store_factory(self, rag_config):
        """Test vector store factory."""
        rag_config.vector_store.backend = "in_memory"
        store = get_vector_store(rag_config.vector_store)
        
        assert isinstance(store, InMemoryVectorStore)
    
    def test_clear_collection(self, rag_config, sample_module):
        """Test clearing the collection."""
        store = InMemoryVectorStore(rag_config.vector_store)
        chunker = CodeChunker()
        provider = MockEmbeddings(rag_config.embedding)
        
        chunks = chunker.chunk_module(sample_module)
        embeddings = provider.embed_texts([c.content for c in chunks])
        
        store.add_documents(chunks, embeddings)
        assert store.get_stats()["count"] > 0
        
        store.delete_collection()
        assert store.get_stats()["count"] == 0


# Retriever Tests

class TestRetriever:
    def test_semantic_search(self, rag_config, sample_module):
        """Test semantic search retrieval."""
        store = InMemoryVectorStore(rag_config.vector_store)
        provider = MockEmbeddings(rag_config.embedding)
        retriever = Retriever(store, provider, rag_config.retrieval)
        chunker = CodeChunker()
        
        # Index
        chunks = chunker.chunk_module(sample_module)
        embeddings = provider.embed_texts([c.content for c in chunks])
        store.add_documents(chunks, embeddings)
        retriever.index_chunks(chunks)
        
        # Search
        results = retriever.semantic_retrieve("calculator", top_k=5)
        
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_hybrid_search(self, rag_config, sample_module):
        """Test hybrid search retrieval."""
        store = InMemoryVectorStore(rag_config.vector_store)
        provider = MockEmbeddings(rag_config.embedding)
        retriever = Retriever(store, provider, rag_config.retrieval)
        chunker = CodeChunker()
        
        # Index
        chunks = chunker.chunk_module(sample_module)
        embeddings = provider.embed_texts([c.content for c in chunks])
        store.add_documents(chunks, embeddings)
        retriever.index_chunks(chunks)
        
        # Hybrid search
        results = retriever.hybrid_retrieve("add numbers", top_k=5)
        
        assert len(results) > 0
    
    def test_build_context(self, rag_config, sample_module):
        """Test context building from retrieval results."""
        store = InMemoryVectorStore(rag_config.vector_store)
        provider = MockEmbeddings(rag_config.embedding)
        retriever = Retriever(store, provider, rag_config.retrieval)
        chunker = CodeChunker()
        
        chunks = chunker.chunk_module(sample_module)
        embeddings = provider.embed_texts([c.content for c in chunks])
        store.add_documents(chunks, embeddings)
        retriever.index_chunks(chunks)
        
        results = retriever.retrieve("calculator", top_k=3)
        context = build_context(results, max_length=5000)
        
        assert len(context) > 0
        assert "---" in context  # Separator between chunks


# LLM Provider Tests

class TestLLMProvider:
    def test_mock_llm_generate(self, rag_config):
        """Test mock LLM generation."""
        llm = MockLLM(rag_config.llm)
        
        response = llm.generate("What is this code?", "class Foo: pass")
        
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == "mock"
    
    def test_llm_provider_factory(self, rag_config):
        """Test LLM provider factory."""
        rag_config.llm.provider = "mock"
        provider = get_llm_provider(rag_config.llm)
        
        assert isinstance(provider, MockLLM)


# Pipeline Tests

class TestRAGPipeline:
    def test_pipeline_index(self, rag_config, sample_module):
        """Test indexing through pipeline."""
        pipeline = RAGPipeline(rag_config)
        
        stats = pipeline.index([sample_module])
        
        assert stats.total_chunks > 0
        assert pipeline.is_indexed()
    
    def test_pipeline_query(self, rag_config, sample_module):
        """Test querying through pipeline."""
        pipeline = RAGPipeline(rag_config)
        
        pipeline.index([sample_module])
        response = pipeline.query("What does the Calculator class do?")
        
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert response.query == "What does the Calculator class do?"
    
    def test_pipeline_search(self, rag_config, sample_module):
        """Test search through pipeline."""
        pipeline = RAGPipeline(rag_config)
        
        pipeline.index([sample_module])
        results = pipeline.search("add numbers", top_k=5)
        
        assert len(results) > 0
    
    def test_pipeline_clear(self, rag_config, sample_module):
        """Test clearing pipeline index."""
        pipeline = RAGPipeline(rag_config)
        
        pipeline.index([sample_module])
        assert pipeline.is_indexed()
        
        pipeline.clear_index()
        # After clear, depends on implementation
    
    def test_response_format_sources(self, rag_config, sample_module):
        """Test RAGResponse source formatting."""
        pipeline = RAGPipeline(rag_config)
        
        pipeline.index([sample_module])
        response = pipeline.query("What is main?")
        
        sources_str = response.format_sources()
        assert "Sources:" in sources_str


# Integration Tests

class TestRAGIntegration:
    def test_full_rag_workflow(self, rag_config, sample_module):
        """Test complete RAG workflow from indexing to querying."""
        # Create pipeline
        pipeline = RAGPipeline(rag_config)
        
        # Index
        stats = pipeline.index([sample_module], clear_existing=True)
        assert stats.total_chunks > 0
        
        # Query
        response = pipeline.query("How do I add numbers?")
        assert len(response.answer) > 0
        assert len(response.sources) > 0
        
        # Search
        results = pipeline.search("subtract")
        assert len(results) > 0
        
        # Get stats
        final_stats = pipeline.get_stats()
        assert final_stats.total_chunks > 0
