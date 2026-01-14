"""Embedding providers for RAG.

Provides abstraction layer for generating vector embeddings from text,
with support for multiple backends including OpenAI, sentence-transformers,
and mock providers for testing.
"""

import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

from analyzer.rag.config import EmbeddingConfig
from analyzer.logging_config import get_logger

logger = get_logger("rag.embeddings")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([query])
        return embeddings[0]
    
    def embed_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """Generate embeddings in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Embedded batch {i // batch_size + 1}")
        return all_embeddings


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider.
    
    Uses OpenAI's text-embedding models for high-quality embeddings.
    Requires OPENAI_API_KEY environment variable.
    """
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = config.model
        self._dimension = self.DIMENSIONS.get(self.model, config.dimension)
        self._client = None
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _get_client(self):
        """Lazily initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []
        
        client = self._get_client()
        
        # Clean texts (OpenAI has limits on empty strings)
        cleaned_texts = [t.replace("\n", " ").strip() or " " for t in texts]
        
        try:
            response = client.embeddings.create(
                model=self.model,
                input=cleaned_texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.
    
    Uses HuggingFace sentence-transformers for local, free embeddings.
    Good for offline use and when API costs are a concern.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_name = config.local_model
        self._model = None
        self._dimension = config.local_dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _get_model(self):
        """Lazily initialize the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Update dimension from loaded model
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers package not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        if not texts:
            return []
        
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing.
    
    Generates deterministic embeddings based on text hash.
    Useful for testing without API calls.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dimension = config.dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings."""
        embeddings = []
        for text in texts:
            embedding = self._generate_deterministic_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def _generate_deterministic_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash."""
        # Use hash to generate consistent "random" values
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Generate embedding values from hash
        embedding = []
        for i in range(self._dimension):
            # Use different parts of the hash
            idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            value = (hash_bytes[idx] / 127.5) - 1
            embedding.append(value)
        
        # Normalize the vector
        norm = sum(x ** 2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


def get_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to get embedding provider based on config.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Configured embedding provider
        
    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = config.provider.lower()
    
    if provider_type == "openai":
        # Check if API key is available
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY not set, falling back to mock embeddings"
            )
            return MockEmbeddings(config)
        return OpenAIEmbeddings(config)
    
    elif provider_type == "sentence-transformers":
        try:
            return SentenceTransformerEmbeddings(config)
        except ImportError:
            logger.warning(
                "sentence-transformers not available, falling back to mock"
            )
            return MockEmbeddings(config)
    
    elif provider_type == "mock":
        return MockEmbeddings(config)
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: list[list[float]]
    texts: list[str]
    dimension: int
    provider: str
    
    def __len__(self) -> int:
        return len(self.embeddings)
