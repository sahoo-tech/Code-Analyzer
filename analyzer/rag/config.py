

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EmbeddingProviderType(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    MOCK = "mock"


class LLMProviderType(Enum):
    """Supported LLM providers."""
    AUTO = "auto"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"
    MOCK = "mock"


class VectorStoreType(Enum):
    """Supported vector store backends."""
    CHROMADB = "chromadb"
    IN_MEMORY = "in_memory"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    # For sentence-transformers
    local_model: str = "all-MiniLM-L6-v2"
    local_dimension: int = 384


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    backend: str = "chromadb"
    persist_directory: str = ".analyzer_rag"
    collection_name: str = "code_entities"
    # Distance metric: "cosine", "l2", "ip" (inner product)
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    """Configuration for code chunking."""
    # Maximum characters per chunk
    chunk_size: int = 1500
    # Overlap between chunks
    chunk_overlap: int = 200
    # Include source code in chunks
    include_source: bool = True
    # Include docstrings
    include_docstrings: bool = True
    # Include metadata (imports, decorators, etc.)
    include_metadata: bool = True
    # Chunk by entity (class/function) boundaries
    entity_based_chunking: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    # Number of results to retrieve
    top_k: int = 10
    # Minimum similarity threshold (0-1 for cosine)
    similarity_threshold: float = 0.5
    # Use hybrid search (semantic + keyword)
    use_hybrid_search: bool = True
    # Weight for semantic search in hybrid mode (0-1)
    semantic_weight: float = 0.7
    # Enable reranking for better relevance
    use_reranking: bool = True
    # Maximum context length for LLM (in characters)
    max_context_length: int = 12000


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "auto"  # "auto", "openai", "anthropic", "google", "gemini", "mock"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    # For Anthropic
    anthropic_model: str = "claude-3-haiku-20240307"
    # For Google
    google_model: str = "gemini-1.5-flash"
    # API timeout in seconds
    timeout: int = 60
    # Enable streaming responses
    streaming: bool = True


@dataclass
class RAGConfig:
    """Main RAG configuration aggregating all sub-configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Global settings
    enabled: bool = True
    # Auto-index on analysis
    auto_index: bool = False
    # Debug mode for verbose logging
    debug: bool = False
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimension": self.embedding.dimension,
                "batch_size": self.embedding.batch_size,
            },
            "vector_store": {
                "backend": self.vector_store.backend,
                "persist_directory": self.vector_store.persist_directory,
                "collection_name": self.vector_store.collection_name,
                "distance_metric": self.vector_store.distance_metric,
            },
            "chunking": {
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "include_source": self.chunking.include_source,
                "entity_based_chunking": self.chunking.entity_based_chunking,
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "similarity_threshold": self.retrieval.similarity_threshold,
                "use_hybrid_search": self.retrieval.use_hybrid_search,
                "use_reranking": self.retrieval.use_reranking,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "enabled": self.enabled,
            "auto_index": self.auto_index,
            "debug": self.debug,
        }


def get_default_rag_config() -> RAGConfig:
    """Get default RAG configuration."""
    return RAGConfig()


def create_rag_config(
    embedding_provider: str = "openai",
    llm_provider: str = "openai",
    persist_directory: str = ".analyzer_rag",
    **kwargs
) -> RAGConfig:

    config = RAGConfig()
    config.embedding.provider = embedding_provider
    config.llm.provider = llm_provider
    config.vector_store.persist_directory = persist_directory
    
    # Set appropriate embedding dimensions based on provider
    if embedding_provider == "sentence-transformers":
        config.embedding.dimension = config.embedding.local_dimension
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
