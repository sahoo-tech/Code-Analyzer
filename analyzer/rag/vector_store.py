"""Vector store implementations for RAG.

Provides abstraction layer for storing and searching vector embeddings,
with support for ChromaDB (persistent) and in-memory stores.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path

from analyzer.rag.config import VectorStoreConfig
from analyzer.rag.chunker import CodeChunk
from analyzer.logging_config import get_logger

logger = get_logger("rag.vector_store")


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    chunk_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    
    # Reconstructed chunk info
    file_path: Optional[str] = None
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    
    def to_chunk(self) -> CodeChunk:
        """Reconstruct CodeChunk from search result."""
        return CodeChunk(
            content=self.content,
            chunk_id=self.chunk_id,
            file_path=self.metadata.get("file_path", ""),
            module_name=self.metadata.get("module_name", ""),
            entity_type=self.metadata.get("entity_type", "unknown"),
            entity_name=self.metadata.get("entity_name", ""),
            start_line=self.metadata.get("start_line", 0),
            end_line=self.metadata.get("end_line", 0),
            parent_name=self.metadata.get("parent_name"),
            metadata=self.metadata,
        )


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(
        self, 
        chunks: list[CodeChunk], 
        embeddings: list[list[float]]
    ) -> None:
        """Add documents with their embeddings to the store.
        
        Args:
            chunks: Code chunks with metadata
            embeddings: Corresponding embedding vectors
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: list[float], 
        top_k: int = 10,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection/index."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the store."""
        pass
    
    def search_by_text(
        self,
        query: str,
        embedding_fn,
        top_k: int = 10
    ) -> list[SearchResult]:
        """Search using text query.
        
        Args:
            query: Text query
            embedding_fn: Function to generate embeddings
            top_k: Number of results
            
        Returns:
            Search results
        """
        query_embedding = embedding_fn(query)
        return self.search(query_embedding, top_k)


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store with persistence.
    
    Uses ChromaDB for efficient similarity search with optional
    persistence to disk.
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
        self._collection = None
    
    def _get_client(self):
        """Lazily initialize ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                persist_dir = Path(self.config.persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                self._client = chromadb.PersistentClient(
                    path=str(persist_dir),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )
                logger.info(f"ChromaDB initialized at {persist_dir}")
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. Install with: pip install chromadb"
                )
        return self._client
    
    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is None:
            client = self._get_client()
            
            # Map distance metrics
            distance_fn = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip",
            }.get(self.config.distance_metric, "cosine")
            
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": distance_fn}
            )
            logger.info(
                f"Using collection '{self.config.collection_name}' "
                f"with {self._collection.count()} documents"
            )
        return self._collection
    
    def add_documents(
        self, 
        chunks: list[CodeChunk], 
        embeddings: list[list[float]]
    ) -> None:
        """Add documents to ChromaDB."""
        if not chunks:
            return
        
        collection = self._get_collection()
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                "file_path": chunk.file_path,
                "module_name": chunk.module_name,
                "entity_type": chunk.entity_type,
                "entity_name": chunk.entity_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            if chunk.parent_name:
                metadata["parent_name"] = chunk.parent_name
            # Add custom metadata (ensure all values are strings/ints/floats)
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
            metadatas.append(metadata)
        
        # Upsert to handle duplicates
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(chunks)} documents to vector store")
    
    def search(
        self, 
        query_embedding: list[float], 
        top_k: int = 10,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents."""
        collection = self._get_collection()
        
        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = filter_metadata
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                # ChromaDB returns distances, lower is better
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine: similarity = 1 - distance
                score = max(0, 1 - distance)
                
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                    file_path=metadata.get("file_path"),
                    entity_type=metadata.get("entity_type"),
                    entity_name=metadata.get("entity_name"),
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                ))
        
        return search_results
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.config.collection_name}'")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        collection = self._get_collection()
        return {
            "name": self.config.collection_name,
            "count": collection.count(),
            "persist_directory": self.config.persist_directory,
        }


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for testing.
    
    Simple implementation using linear search.
    Not suitable for large datasets.
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._documents: list[tuple[CodeChunk, list[float]]] = []
    
    def add_documents(
        self, 
        chunks: list[CodeChunk], 
        embeddings: list[list[float]]
    ) -> None:
        """Add documents to memory."""
        for chunk, embedding in zip(chunks, embeddings):
            self._documents.append((chunk, embedding))
        logger.info(f"Added {len(chunks)} documents to memory store")
    
    def search(
        self, 
        query_embedding: list[float], 
        top_k: int = 10,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search using cosine similarity."""
        if not self._documents:
            return []
        
        # Calculate similarities
        scored_results = []
        for chunk, embedding in self._documents:
            score = self._cosine_similarity(query_embedding, embedding)
            scored_results.append((chunk, score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for chunk, score in scored_results[:top_k]:
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=score,
                metadata=chunk.to_dict(),
                file_path=chunk.file_path,
                entity_type=chunk.entity_type,
                entity_name=chunk.entity_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            ))
        
        return results
    
    def delete_collection(self) -> None:
        """Clear all documents."""
        self._documents = []
        logger.info("Cleared in-memory store")
    
    def get_stats(self) -> dict:
        """Get store statistics."""
        return {
            "name": "in_memory",
            "count": len(self._documents),
        }
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def get_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Factory function to get vector store based on config.
    
    Args:
        config: Vector store configuration
        
    Returns:
        Configured vector store
        
    Raises:
        ValueError: If backend type is not supported
    """
    backend = config.backend.lower()
    
    if backend == "chromadb":
        try:
            return ChromaVectorStore(config)
        except ImportError:
            logger.warning("ChromaDB not available, using in-memory store")
            return InMemoryVectorStore(config)
    
    elif backend == "in_memory":
        return InMemoryVectorStore(config)
    
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
