"""
FastAPI server for RAG system.
Provides REST API endpoints for document management and querying.
"""

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn

from .rag_system import RAGSystem
from .multi_hop_agent import MultiHopAgent
from .visualizations import CostVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Cost-Optimized System API",
    description="API for cost-optimized RAG with vector database and caching",
    version="1.0.0"
)

# Initialize RAG system
rag_system = RAGSystem()
multi_hop_agent = MultiHopAgent(rag_system)
visualizer = CostVisualizer()


# Pydantic models
class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None


class DocumentBatch(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict]] = None
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class Query(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True
    generate_answer: bool = True


class MultiHopQuery(BaseModel):
    query: str
    max_hops: int = Field(default=3, ge=1, le=5)
    use_cache: bool = True


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Cost-Optimized System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /documents": "Add documents to vector store",
            "POST /query": "Query the RAG system",
            "POST /query/multi-hop": "Execute multi-hop reasoning query",
            "GET /stats": "Get system statistics",
            "GET /costs": "Get cost summary",
            "POST /costs/reset": "Reset cost tracking",
            "POST /cache/clear": "Clear cache",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cache_enabled": rag_system.cache.enabled,
        "vector_store_docs": rag_system.vector_store.count()
    }


@app.post("/documents")
async def add_documents(batch: DocumentBatch):
    """
    Add documents to the vector store.
    
    Returns operation results including cost.
    """
    try:
        result = rag_system.add_documents(
            documents=batch.documents,
            metadatas=batch.metadatas,
            chunk_size=batch.chunk_size,
            chunk_overlap=batch.chunk_overlap
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_rag(query: Query):
    """
    Query the RAG system.
    
    Returns answer, retrieved documents, and cost information.
    """
    try:
        result = rag_system.query(
            query=query.query,
            top_k=query.top_k,
            use_cache=query.use_cache,
            generate_answer=query.generate_answer
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/multi-hop")
async def multi_hop_query(query: MultiHopQuery):
    """
    Execute multi-hop reasoning query.
    
    Returns answer with reasoning steps and cost breakdown.
    """
    try:
        # Update max_hops if specified
        multi_hop_agent.max_hops = query.max_hops
        
        result = multi_hop_agent.query(
            query=query.query,
            use_cache=query.use_cache
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error processing multi-hop query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = rag_system.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/costs")
async def get_costs():
    """Get detailed cost summary."""
    try:
        summary = rag_system.get_cost_summary()
        return {
            "success": True,
            "costs": summary
        }
    except Exception as e:
        logger.error(f"Error getting costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/costs/reset")
async def reset_costs():
    """Reset cost tracking."""
    try:
        rag_system.reset_costs()
        return {
            "success": True,
            "message": "Cost tracking reset"
        }
    except Exception as e:
        logger.error(f"Error resetting costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear the cache."""
    try:
        rag_system.clear_cache()
        return {
            "success": True,
            "message": "Cache cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/generate")
async def generate_visualizations(background_tasks: BackgroundTasks):
    """Generate cost visualization plots."""
    try:
        cost_summary = rag_system.get_cost_summary()
        
        # Generate plots in background
        def generate_plots():
            visualizer.generate_all_plots(cost_summary)
        
        background_tasks.add_task(generate_plots)
        
        return {
            "success": True,
            "message": "Visualization generation started",
            "output_dir": visualizer.output_dir
        }
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
