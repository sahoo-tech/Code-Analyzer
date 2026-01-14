"""Main RAG pipeline orchestrator.

Coordinates all RAG components: chunking, embedding, indexing,
retrieval, and LLM-based answer generation.
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from pathlib import Path

from analyzer.models.code_entities import Module
from analyzer.rag.config import RAGConfig, get_default_rag_config
from analyzer.rag.chunker import CodeChunker, CodeChunk
from analyzer.rag.embeddings import get_embedding_provider, EmbeddingProvider
from analyzer.rag.vector_store import get_vector_store, VectorStore, SearchResult
from analyzer.rag.retriever import Retriever, RetrievalResult, build_context
from analyzer.rag.llm_provider import get_llm_provider, LLMProvider, LLMResponse
from analyzer.rag.prompts import get_prompt_template, format_prompt
from analyzer.logging_config import get_logger

logger = get_logger("rag.pipeline")


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: list[CodeChunk]
    query: str
    retrieval_scores: list[float]
    llm_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "retrieval_scores": self.retrieval_scores,
            "model": self.llm_response.model if self.llm_response else None,
        }
    
    def format_sources(self) -> str:
        """Format sources for display."""
        if not self.sources:
            return "No sources found."
        
        lines = ["Sources:"]
        for i, source in enumerate(self.sources, 1):
            score = self.retrieval_scores[i-1] if i <= len(self.retrieval_scores) else 0
            lines.append(
                f"  {i}. [{source.entity_type}] {source.full_name} "
                f"(score: {score:.2f})"
            )
            lines.append(f"     File: {source.file_path}:{source.start_line}")
        
        return "\n".join(lines)


@dataclass
class IndexStats:
    """Statistics about the RAG index."""
    total_chunks: int = 0
    total_modules: int = 0
    total_classes: int = 0
    total_functions: int = 0
    total_methods: int = 0
    persist_directory: str = ""
    embedding_provider: str = ""
    embedding_dimension: int = 0
    
    def to_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "total_modules": self.total_modules,
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_methods": self.total_methods,
            "persist_directory": self.persist_directory,
            "embedding_provider": self.embedding_provider,
            "embedding_dimension": self.embedding_dimension,
        }


class RAGPipeline:
    """Main RAG pipeline orchestrating all components.
    
    Provides:
    - Indexing: Chunk, embed, and store code modules
    - Query: Retrieve relevant code and generate answers
    - Search: Semantic search without LLM generation
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_default_rag_config()
        
        # Initialize components lazily
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._chunker: Optional[CodeChunker] = None
        
        # Index state
        self._indexed = False
        self._chunks: list[CodeChunk] = []
        self._stats = IndexStats()
    
    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(self.config.embedding)
            logger.info(
                f"Initialized embedding provider: {self.config.embedding.provider}"
            )
        return self._embedding_provider
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.config.vector_store)
            logger.info(
                f"Initialized vector store: {self.config.vector_store.backend}"
            )
        return self._vector_store
    
    @property
    def retriever(self) -> Retriever:
        """Get or create retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                self.vector_store,
                self.embedding_provider,
                self.config.retrieval,
            )
        return self._retriever
    
    @property
    def llm_provider(self) -> LLMProvider:
        """Get or create LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(self.config.llm)
            logger.info(f"Initialized LLM provider: {self.config.llm.provider}")
        return self._llm_provider
    
    @property
    def chunker(self) -> CodeChunker:
        """Get or create chunker."""
        if self._chunker is None:
            self._chunker = CodeChunker(self.config.chunking)
        return self._chunker
    
    def index(
        self,
        modules: list[Module],
        project_path: Optional[str] = None,
        clear_existing: bool = False,
    ) -> IndexStats:
        """Index code modules for RAG.
        
        Args:
            modules: Parsed code modules to index
            project_path: Optional project path for context
            clear_existing: Whether to clear existing index
            
        Returns:
            Index statistics
        """
        if clear_existing:
            self.clear_index()
        
        logger.info(f"Indexing {len(modules)} modules for RAG")
        
        # Chunk all modules
        chunks = self.chunker.chunk_modules(modules)
        self._chunks = chunks
        
        if not chunks:
            logger.warning("No chunks created from modules")
            return self._stats
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed_batch(
            texts, 
            batch_size=self.config.embedding.batch_size
        )
        
        # Store in vector database
        logger.info("Storing embeddings in vector store")
        self.vector_store.add_documents(chunks, embeddings)
        
        # Index for keyword search
        self.retriever.index_chunks(chunks)
        
        # Update stats
        self._stats = IndexStats(
            total_chunks=len(chunks),
            total_modules=sum(1 for c in chunks if c.entity_type == "module"),
            total_classes=sum(1 for c in chunks if c.entity_type == "class"),
            total_functions=sum(1 for c in chunks if c.entity_type == "function"),
            total_methods=sum(1 for c in chunks if c.entity_type == "method"),
            persist_directory=self.config.vector_store.persist_directory,
            embedding_provider=self.config.embedding.provider,
            embedding_dimension=self.embedding_provider.dimension,
        )
        
        self._indexed = True
        logger.info(f"Indexing complete: {self._stats.total_chunks} chunks indexed")
        
        return self._stats
    
    def query(
        self,
        question: str,
        prompt_type: str = "qa",
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Query the indexed codebase.
        
        Args:
            question: Natural language question
            prompt_type: Type of prompt template to use
            top_k: Number of chunks to retrieve
            
        Returns:
            RAG response with answer and sources
        """
        top_k = top_k or self.config.retrieval.top_k
        
        # Retrieve relevant chunks
        logger.info(f"Retrieving relevant chunks for: {question[:50]}...")
        results = self.retriever.retrieve(question, top_k=top_k)
        
        if not results:
            return RAGResponse(
                answer="No relevant code found for your question. "
                       "Make sure the codebase is indexed.",
                sources=[],
                query=question,
                retrieval_scores=[],
            )
        
        # Build context
        context = build_context(
            results,
            max_length=self.config.retrieval.max_context_length,
        )
        
        # Generate answer
        logger.info("Generating answer with LLM")
        template = get_prompt_template(prompt_type)
        prompt = format_prompt(template, question=question, context=context)
        
        llm_response = self.llm_provider.generate(prompt, "")
        
        return RAGResponse(
            answer=llm_response.content,
            sources=[r.chunk for r in results],
            query=question,
            retrieval_scores=[r.score for r in results],
            llm_response=llm_response,
        )
    
    async def query_stream(
        self,
        question: str,
        prompt_type: str = "qa",
        top_k: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream query response.
        
        Args:
            question: Natural language question
            prompt_type: Type of prompt
            top_k: Number of chunks
            
        Yields:
            Response chunks
        """
        top_k = top_k or self.config.retrieval.top_k
        
        # Retrieve
        results = self.retriever.retrieve(question, top_k=top_k)
        
        if not results:
            yield "No relevant code found for your question."
            return
        
        # Build context
        context = build_context(
            results,
            max_length=self.config.retrieval.max_context_length,
        )
        
        # Stream response
        template = get_prompt_template(prompt_type)
        prompt = format_prompt(template, question=question, context=context)
        
        async for chunk in self.llm_provider.generate_stream(prompt, ""):
            yield chunk
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_entity_type: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Semantic search without LLM generation.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_entity_type: Optional entity type filter
            
        Returns:
            List of retrieval results
        """
        top_k = top_k or self.config.retrieval.top_k
        
        filter_metadata = None
        if filter_entity_type:
            filter_metadata = {"entity_type": filter_entity_type}
        
        return self.retriever.retrieve(
            query, 
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
    
    def get_stats(self) -> IndexStats:
        """Get current index statistics."""
        # Update from vector store
        store_stats = self.vector_store.get_stats()
        self._stats.total_chunks = store_stats.get("count", 0)
        return self._stats
    
    def clear_index(self) -> None:
        """Clear the entire index."""
        self.vector_store.delete_collection()
        self._chunks = []
        self._indexed = False
        self._stats = IndexStats()
        logger.info("Index cleared")
    
    def is_indexed(self) -> bool:
        """Check if any documents are indexed."""
        stats = self.vector_store.get_stats()
        return stats.get("count", 0) > 0


def create_rag_pipeline(
    embedding_provider: str = "openai",
    llm_provider: str = "openai",
    persist_directory: str = ".analyzer_rag",
) -> RAGPipeline:
    """Create a RAG pipeline with common settings.
    
    Args:
        embedding_provider: Embedding provider to use
        llm_provider: LLM provider to use
        persist_directory: Directory for vector store
        
    Returns:
        Configured RAG pipeline
    """
    from analyzer.rag.config import create_rag_config
    
    config = create_rag_config(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        persist_directory=persist_directory,
    )
    
    return RAGPipeline(config)
