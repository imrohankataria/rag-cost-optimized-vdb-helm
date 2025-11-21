"""
Core RAG system implementation with cost tracking and caching.
"""

import logging
import os
from typing import List, Dict, Optional
from openai import OpenAI

from .cost_tracker import CostTracker, get_global_tracker
from .cache_layer import CacheLayer
from .vector_store import VectorStore, chunk_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system with integrated cost tracking and caching."""
    
    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 enable_cache: bool = True,
                 cache_ttl: int = 3600,
                 collection_name: str = "rag_documents",
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize RAG system.
        
        Args:
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            enable_cache: Enable caching layer
            cache_ttl: Cache TTL in seconds
            collection_name: ChromaDB collection name
            embedding_model: OpenAI embedding model
            llm_model: OpenAI LLM model
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key provided. Some features will not work.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize components
        self.cost_tracker = get_global_tracker()
        self.cache = CacheLayer(enabled=enable_cache, ttl=cache_ttl)
        self.vector_store = VectorStore(collection_name=collection_name)
        
        logger.info("RAG system initialized")
    
    def add_documents(self,
                     documents: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> Dict:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary with operation results
        """
        logger.info(f"Adding {len(documents)} documents")
        
        all_chunks = []
        all_metadatas = []
        chunk_costs = 0.0
        
        for i, doc in enumerate(documents):
            # Chunk document
            chunks = chunk_text(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            
            # Create metadata for each chunk
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            for j, chunk in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "doc_id": i,
                    "chunk_id": j,
                    "chunk_count": len(chunks)
                }
                all_metadatas.append(chunk_metadata)
                
                # Track embedding cost (ChromaDB handles embeddings internally)
                chunk_costs += self.cost_tracker.track_embedding(
                    chunk, 
                    self.embedding_model,
                    cached=False
                )
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        result = {
            "documents_added": len(documents),
            "chunks_created": len(all_chunks),
            "doc_ids": doc_ids,
            "cost": chunk_costs
        }
        
        logger.info(f"Added {len(all_chunks)} chunks. Cost: ${chunk_costs:.4f}")
        return result
    
    def query(self,
             query: str,
             top_k: int = 5,
             use_cache: bool = True,
             generate_answer: bool = True) -> Dict:
        """
        Query the RAG system.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            use_cache: Use caching for retrieval
            generate_answer: Generate answer using LLM
            
        Returns:
            Dictionary with answer, documents, and costs
        """
        logger.info(f"Processing query: {query}")
        
        query_cost = 0.0
        cached = False
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get_retrieval(query)
            if cached_result:
                logger.info("Retrieved from cache")
                cached = True
                documents = cached_result["documents"]
                
                # Track cached retrieval cost (minimal)
                query_cost += self.cost_tracker.track_vector_search(cached=True)
                query_cost += self.cost_tracker.track_embedding(
                    query,
                    self.embedding_model,
                    cached=True
                )
            else:
                documents = None
        else:
            documents = None
        
        # Retrieve from vector store if not cached
        if documents is None:
            # Track query embedding cost
            query_cost += self.cost_tracker.track_embedding(
                query,
                self.embedding_model,
                cached=False
            )
            
            # Track vector search cost
            query_cost += self.cost_tracker.track_vector_search(cached=False)
            
            # Retrieve documents
            results = self.vector_store.query(query, n_results=top_k)
            documents = results["documents"]
            
            # Cache results
            if use_cache:
                self.cache.cache_retrieval(query, documents)
        
        result = {
            "query": query,
            "documents": documents,
            "cached": cached,
            "cost": query_cost
        }
        
        # Generate answer if requested
        if generate_answer and documents:
            answer, answer_cost = self._generate_answer(query, documents)
            result["answer"] = answer
            result["cost"] += answer_cost
        else:
            result["answer"] = None
        
        logger.info(f"Query completed. Cost: ${result['cost']:.4f}, Cached: {cached}")
        return result
    
    def _generate_answer(self, query: str, documents: List[str]) -> tuple[str, float]:
        """
        Generate answer using LLM.
        
        Returns:
            Tuple of (answer, cost)
        """
        if not self.client:
            logger.warning("No OpenAI client available")
            return "OpenAI client not configured", 0.0
        
        # Prepare context from documents
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents[:3])])
        
        # Create prompt
        prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Answer:"""
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Track cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = self.cost_tracker.track_llm_generation(
                prompt_tokens,
                completion_tokens,
                self.llm_model
            )
            
            return answer, cost
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}", 0.0
    
    def get_cost_summary(self) -> Dict:
        """Get summary of all costs."""
        summary = self.cost_tracker.get_summary()
        summary["cache_stats"] = self.cache.get_stats()
        summary["cache_savings"] = self.cost_tracker.get_cache_savings()
        return summary
    
    def reset_costs(self):
        """Reset cost tracking."""
        self.cost_tracker.reset()
        self.cache.reset_stats()
        logger.info("Cost tracking reset")
    
    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "cache": self.cache.get_stats(),
            "costs": self.cost_tracker.get_summary()
        }


if __name__ == "__main__":
    # Simple test
    rag = RAGSystem(enable_cache=False)
    
    # Add sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and humans using natural language."
    ]
    
    result = rag.add_documents(sample_docs)
    print(f"Added documents. Cost: ${result['cost']:.4f}")
    
    # Query
    query_result = rag.query("What is machine learning?", generate_answer=False)
    print(f"\nQuery: {query_result['query']}")
    print(f"Retrieved {len(query_result['documents'])} documents")
    print(f"Cost: ${query_result['cost']:.4f}")
    print(f"\nCost Summary:")
    print(rag.get_cost_summary())
