"""
Vector store operations using ChromaDB.
Handles document storage, embedding, and retrieval.
"""

import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for document embeddings."""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str],
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "default"} for _ in documents]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(self, 
             query_text: str,
             n_results: int = 5,
             where: Optional[Dict] = None) -> Dict:
        """
        Query vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
            logger.info(f"Retrieved {len(formatted_results['documents'])} documents for query")
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """Get documents by their IDs."""
        try:
            results = self.collection.get(ids=ids)
            logger.info(f"Retrieved {len(results['documents'])} documents by ID")
            return results
        except Exception as e:
            logger.error(f"Error getting documents by ID: {e}")
            raise
    
    def delete_documents(self, ids: List[str]):
        """Delete documents from vector store."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict] = None):
        """Update a document in vector store."""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata] if metadata else None
            )
            logger.info(f"Updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    def count(self) -> int:
        """Get total number of documents in collection."""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def clear(self):
        """Clear all documents from collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            return {
                "name": self.collection_name,
                "count": self.count(),
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks
