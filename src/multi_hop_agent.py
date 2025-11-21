"""
Multi-hop reasoning agent for complex queries.
Implements iterative query refinement and multiple retrieval steps.
"""

import logging
from typing import Dict, List, Optional
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class MultiHopAgent:
    """Agent for multi-hop reasoning over retrieved documents."""
    
    def __init__(self, rag_system, max_hops: int = 3):
        """
        Initialize multi-hop agent.
        
        Args:
            rag_system: RAG system instance to use for retrieval
            max_hops: Maximum number of reasoning hops
        """
        self.rag_system = rag_system
        self.max_hops = max_hops
        self.cost_tracker = CostTracker()
    
    def query(self, 
             query: str,
             use_cache: bool = True) -> Dict:
        """
        Execute multi-hop query with iterative refinement.
        
        Args:
            query: Complex query requiring multiple steps
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with answer, reasoning steps, and costs
        """
        logger.info(f"Starting multi-hop query: {query}")
        
        reasoning_steps = []
        all_documents = []
        hop_count = 0
        
        current_query = query
        
        # First hop: Initial retrieval
        hop_count += 1
        logger.info(f"Hop {hop_count}: Initial retrieval")
        
        result = self.rag_system.query(current_query, use_cache=use_cache)
        
        reasoning_steps.append({
            "hop": hop_count,
            "query": current_query,
            "documents_retrieved": len(result.get("documents", [])),
            "cost": result.get("cost", 0.0)
        })
        
        all_documents.extend(result.get("documents", []))
        initial_answer = result.get("answer", "")
        
        # Check if we need additional hops
        # Simple heuristic: if answer is short or mentions "more information needed"
        needs_more_info = (
            len(initial_answer) < 100 or
            "more information" in initial_answer.lower() or
            "unclear" in initial_answer.lower()
        )
        
        # Additional hops if needed
        while needs_more_info and hop_count < self.max_hops:
            hop_count += 1
            logger.info(f"Hop {hop_count}: Refinement")
            
            # Generate refinement query based on previous answer
            refinement_query = self._generate_refinement_query(
                original_query=query,
                previous_answer=initial_answer,
                hop=hop_count
            )
            
            # Retrieve with refined query
            refined_result = self.rag_system.query(
                refinement_query, 
                use_cache=use_cache
            )
            
            reasoning_steps.append({
                "hop": hop_count,
                "query": refinement_query,
                "documents_retrieved": len(refined_result.get("documents", [])),
                "cost": refined_result.get("cost", 0.0)
            })
            
            all_documents.extend(refined_result.get("documents", []))
            
            # Update answer with refined information
            initial_answer = self._synthesize_answer(
                query=query,
                documents=all_documents,
                previous_answer=initial_answer
            )
            
            # Check if we have enough information now
            needs_more_info = (
                len(initial_answer) < 150 and 
                hop_count < self.max_hops
            )
        
        # Calculate total cost
        total_cost = sum(step["cost"] for step in reasoning_steps)
        
        final_result = {
            "answer": initial_answer,
            "reasoning_steps": reasoning_steps,
            "total_hops": hop_count,
            "total_documents": len(all_documents),
            "total_cost": total_cost,
            "documents": all_documents
        }
        
        logger.info(f"Multi-hop query completed in {hop_count} hops. Total cost: ${total_cost:.4f}")
        
        return final_result
    
    def _generate_refinement_query(self, 
                                   original_query: str,
                                   previous_answer: str,
                                   hop: int) -> str:
        """Generate a refined query based on previous results."""
        # Simple refinement strategy
        refinements = [
            f"Provide more details about: {original_query}",
            f"Explain further: {original_query}",
            f"What are additional aspects of: {original_query}"
        ]
        
        # Cycle through refinement strategies
        refinement_index = (hop - 2) % len(refinements)
        return refinements[refinement_index]
    
    def _synthesize_answer(self,
                          query: str,
                          documents: List[str],
                          previous_answer: str) -> str:
        """Synthesize answer from multiple documents."""
        # In a real implementation, this would use an LLM
        # For now, we'll create a simple synthesis
        
        if not documents:
            return previous_answer
        
        # Combine unique information from documents
        doc_snippets = []
        for i, doc in enumerate(documents[-3:]):  # Use last 3 documents
            snippet = doc[:200] if len(doc) > 200 else doc
            doc_snippets.append(snippet)
        
        # Simple synthesis
        synthesized = f"{previous_answer}\n\nAdditional context: {' '.join(doc_snippets[:2])}"
        
        return synthesized
    
    def compare_and_analyze(self,
                           query: str,
                           aspects: List[str]) -> Dict:
        """
        Compare multiple aspects of a topic.
        
        Args:
            query: Base query
            aspects: List of aspects to compare
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing aspects: {aspects}")
        
        comparisons = []
        total_cost = 0.0
        
        for aspect in aspects:
            aspect_query = f"{query} - specifically regarding {aspect}"
            result = self.rag_system.query(aspect_query)
            
            comparisons.append({
                "aspect": aspect,
                "answer": result.get("answer", ""),
                "documents": result.get("documents", []),
                "cost": result.get("cost", 0.0)
            })
            
            total_cost += result.get("cost", 0.0)
        
        # Synthesize comparison
        comparison_summary = self._create_comparison_summary(query, comparisons)
        
        return {
            "query": query,
            "aspects": aspects,
            "comparisons": comparisons,
            "summary": comparison_summary,
            "total_cost": total_cost
        }
    
    def _create_comparison_summary(self, query: str, comparisons: List[Dict]) -> str:
        """Create a summary comparing different aspects."""
        summary_parts = [f"Comparison of {query}:\n"]
        
        for comp in comparisons:
            aspect = comp["aspect"]
            answer = comp["answer"][:200]  # Truncate
            summary_parts.append(f"\n{aspect}: {answer}")
        
        return "".join(summary_parts)
    
    def get_cost_breakdown(self) -> Dict:
        """Get detailed cost breakdown for multi-hop operations."""
        return self.cost_tracker.get_breakdown()
