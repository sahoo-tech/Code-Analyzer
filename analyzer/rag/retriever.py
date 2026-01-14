"""Retrieval system for RAG.

Implements hybrid retrieval combining semantic search with keyword matching,
plus optional reranking for improved relevance.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from analyzer.rag.config import RetrievalConfig
from analyzer.rag.chunker import CodeChunk
from analyzer.rag.embeddings import EmbeddingProvider
from analyzer.rag.vector_store import VectorStore, SearchResult
from analyzer.logging_config import get_logger

logger = get_logger("rag.retriever")


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk: CodeChunk
    score: float
    source: str  # "semantic", "keyword", "hybrid"
    rank: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "source": self.source,
            "rank": self.rank,
        }


class Retriever:
    """Hybrid retriever combining semantic and keyword search.
    
    Provides multiple retrieval strategies:
    - Semantic search using vector embeddings
    - Keyword search using BM25-like scoring
    - Hybrid combining both approaches
    - Optional reranking for improved relevance
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()
        
        # Cache for keyword search
        self._keyword_index: dict[str, list[tuple[CodeChunk, float]]] = {}
        self._chunks: list[CodeChunk] = []
    
    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """Index chunks for keyword search.
        
        Args:
            chunks: Chunks to index
        """
        self._chunks = chunks
        self._build_keyword_index(chunks)
        logger.info(f"Indexed {len(chunks)} chunks for keyword search")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Uses the configured retrieval strategy (semantic, keyword, or hybrid).
        
        Args:
            query: Query string
            top_k: Number of results (defaults to config)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieval results
        """
        top_k = top_k or self.config.top_k
        
        if self.config.use_hybrid_search:
            return self.hybrid_retrieve(query, top_k, filter_metadata)
        else:
            return self.semantic_retrieve(query, top_k, filter_metadata)
    
    def semantic_retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using semantic similarity only.
        
        Args:
            query: Query string
            top_k: Number of results
            filter_metadata: Optional filters
            
        Returns:
            Retrieval results
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_query(query)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding, 
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        # Convert to retrieval results
        results = []
        for i, sr in enumerate(search_results):
            if sr.score >= self.config.similarity_threshold:
                results.append(RetrievalResult(
                    chunk=sr.to_chunk(),
                    score=sr.score,
                    source="semantic",
                    rank=i + 1,
                ))
        
        return results
    
    def keyword_retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve using keyword matching.
        
        Uses a simplified BM25-like scoring based on term frequency.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            Retrieval results
        """
        if not self._chunks:
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Score each chunk
        scored_chunks = []
        for chunk in self._chunks:
            score = self._keyword_score(chunk, query_terms)
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for i, (chunk, score) in enumerate(scored_chunks[:top_k]):
            results.append(RetrievalResult(
                chunk=chunk,
                score=min(score, 1.0),  # Normalize
                source="keyword",
                rank=i + 1,
            ))
        
        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using both semantic and keyword search.
        
        Combines results using reciprocal rank fusion.
        
        Args:
            query: Query string
            top_k: Number of results
            filter_metadata: Optional filters
            
        Returns:
            Combined retrieval results
        """
        # Get results from both methods
        semantic_results = self.semantic_retrieve(
            query, top_k * 2, filter_metadata
        )
        keyword_results = self.keyword_retrieve(query, top_k * 2)
        
        # Combine using reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            self.config.semantic_weight,
        )
        
        # Optionally rerank
        if self.config.use_reranking:
            combined = self._rerank(combined, query)
        
        # Return top-k
        return combined[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        semantic_weight: float = 0.7,
    ) -> list[RetrievalResult]:
        """Combine results using reciprocal rank fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list.
        """
        k = 60  # RRF constant
        chunk_scores: dict[str, tuple[float, CodeChunk]] = {}
        
        # Score semantic results
        for result in semantic_results:
            chunk_id = result.chunk.chunk_id
            rrf_score = semantic_weight / (k + result.rank)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (
                    chunk_scores[chunk_id][0] + rrf_score,
                    result.chunk,
                )
            else:
                chunk_scores[chunk_id] = (rrf_score, result.chunk)
        
        # Score keyword results
        keyword_weight = 1 - semantic_weight
        for result in keyword_results:
            chunk_id = result.chunk.chunk_id
            rrf_score = keyword_weight / (k + result.rank)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (
                    chunk_scores[chunk_id][0] + rrf_score,
                    result.chunk,
                )
            else:
                chunk_scores[chunk_id] = (rrf_score, result.chunk)
        
        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )
        
        # Create result objects
        results = []
        for i, (chunk_id, (score, chunk)) in enumerate(sorted_results):
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                source="hybrid",
                rank=i + 1,
            ))
        
        return results
    
    def _rerank(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        """Rerank results for improved relevance.
        
        Simple heuristic reranking based on:
        - Exact phrase matches
        - Query term coverage
        - Entity type relevance
        """
        query_lower = query.lower()
        query_terms = set(self._tokenize(query))
        
        reranked = []
        for result in results:
            content_lower = result.chunk.content.lower()
            
            # Boost for exact phrase match
            phrase_boost = 0.3 if query_lower in content_lower else 0
            
            # Boost for term coverage
            content_terms = set(self._tokenize(result.chunk.content))
            coverage = len(query_terms & content_terms) / max(len(query_terms), 1)
            coverage_boost = coverage * 0.2
            
            # Boost for entity name match
            entity_name = result.chunk.entity_name.lower()
            name_boost = 0.2 if any(t in entity_name for t in query_terms) else 0
            
            # Calculate new score
            new_score = result.score + phrase_boost + coverage_boost + name_boost
            
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=new_score,
                source=result.source,
                rank=result.rank,
            ))
        
        # Re-sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        return reranked
    
    def _build_keyword_index(self, chunks: list[CodeChunk]) -> None:
        """Build inverted index for keyword search."""
        self._keyword_index.clear()
        
        for chunk in chunks:
            terms = self._tokenize(chunk.content)
            term_freq = Counter(terms)
            
            for term, freq in term_freq.items():
                if term not in self._keyword_index:
                    self._keyword_index[term] = []
                # TF score
                tf = freq / max(len(terms), 1)
                self._keyword_index[term].append((chunk, tf))
    
    def _keyword_score(
        self,
        chunk: CodeChunk,
        query_terms: list[str],
    ) -> float:
        """Calculate keyword match score for a chunk."""
        content_lower = chunk.content.lower()
        score = 0.0
        matched_terms = 0
        
        for term in query_terms:
            if term in content_lower:
                matched_terms += 1
                # Count occurrences
                count = content_lower.count(term)
                # Log-normalized frequency
                score += (1 + count) / (1 + len(query_terms))
        
        # Boost for matching all terms
        if matched_terms == len(query_terms):
            score *= 1.5
        
        return score
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into terms."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z_][a-z0-9_]*\b', text)
        # Filter short tokens and common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'in', 'it', 'for', 'on', 'with', 'as', 'this', 'that', 'self'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]


def build_context(
    results: list[RetrievalResult],
    max_length: int = 12000,
) -> str:
    """Build context string from retrieval results.
    
    Args:
        results: Retrieval results
        max_length: Maximum context length in characters
        
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for result in results:
        chunk = result.chunk
        
        # Format chunk
        header = f"[{chunk.entity_type.upper()}: {chunk.full_name}]"
        location = f"File: {chunk.file_path}, Lines {chunk.start_line}-{chunk.end_line}"
        content = chunk.content
        
        part = f"{header}\n{location}\n{content}\n"
        
        if current_length + len(part) > max_length:
            # Truncate if needed
            remaining = max_length - current_length - 50
            if remaining > 200:
                part = part[:remaining] + "\n...[truncated]"
                context_parts.append(part)
            break
        
        context_parts.append(part)
        current_length += len(part)
    
    return "\n---\n".join(context_parts)
