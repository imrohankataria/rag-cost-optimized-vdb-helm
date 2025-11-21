"""
Cost tracking module for RAG operations.
Tracks costs for embeddings, vector searches, and LLM generations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class CostTracker:
    """Track and analyze costs for RAG operations."""
    
    # OpenAI pricing (as of 2024)
    PRICING = {
        "text-embedding-ada-002": {
            "input": 0.0001 / 1000  # per token
        },
        "gpt-3.5-turbo": {
            "input": 0.0005 / 1000,  # per token
            "output": 0.0015 / 1000  # per token
        },
        "gpt-4": {
            "input": 0.03 / 1000,
            "output": 0.06 / 1000
        }
    }
    
    def __init__(self):
        self.operations: List[Dict] = []
        self.total_cost = 0.0
        
    def track_embedding(self, text: str, model: str = "text-embedding-ada-002", 
                       cached: bool = False) -> float:
        """Track cost of an embedding operation."""
        # Approximate token count (1 token ~= 4 chars)
        token_count = len(text) / 4
        
        if cached:
            cost = 0.0  # No cost for cached embeddings
        else:
            cost = token_count * self.PRICING[model]["input"]
        
        operation = {
            "type": "embedding",
            "model": model,
            "tokens": token_count,
            "cost": cost,
            "cached": cached,
            "timestamp": datetime.now().isoformat()
        }
        
        self.operations.append(operation)
        self.total_cost += cost
        
        logger.info(f"Embedding cost: ${cost:.6f} (cached: {cached})")
        return cost
    
    def track_llm_generation(self, prompt_tokens: int, completion_tokens: int,
                            model: str = "gpt-3.5-turbo") -> float:
        """Track cost of LLM generation."""
        input_cost = prompt_tokens * self.PRICING[model]["input"]
        output_cost = completion_tokens * self.PRICING[model]["output"]
        total_cost = input_cost + output_cost
        
        operation = {
            "type": "llm_generation",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cost": total_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        self.operations.append(operation)
        self.total_cost += total_cost
        
        logger.info(f"LLM generation cost: ${total_cost:.6f}")
        return total_cost
    
    def track_vector_search(self, cached: bool = False) -> float:
        """Track cost of vector database search."""
        # Vector search has compute cost
        cost = 0.0001 if cached else 0.001
        
        operation = {
            "type": "vector_search",
            "cost": cost,
            "cached": cached,
            "timestamp": datetime.now().isoformat()
        }
        
        self.operations.append(operation)
        self.total_cost += cost
        
        logger.info(f"Vector search cost: ${cost:.6f} (cached: {cached})")
        return cost
    
    def track_custom_operation(self, operation_type: str, cost: float, 
                              metadata: Optional[Dict] = None):
        """Track a custom operation cost."""
        operation = {
            "type": operation_type,
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            operation.update(metadata)
        
        self.operations.append(operation)
        self.total_cost += cost
        
        logger.info(f"{operation_type} cost: ${cost:.6f}")
    
    def get_breakdown(self) -> Dict:
        """Get detailed cost breakdown by operation type."""
        breakdown = {}
        
        for op in self.operations:
            op_type = op["type"]
            if op_type not in breakdown:
                breakdown[op_type] = {
                    "count": 0,
                    "total_cost": 0.0,
                    "cached_count": 0,
                    "uncached_count": 0
                }
            
            breakdown[op_type]["count"] += 1
            breakdown[op_type]["total_cost"] += op["cost"]
            
            if "cached" in op:
                if op["cached"]:
                    breakdown[op_type]["cached_count"] += 1
                else:
                    breakdown[op_type]["uncached_count"] += 1
        
        return breakdown
    
    def get_summary(self) -> Dict:
        """Get summary of all tracked costs."""
        breakdown = self.get_breakdown()
        
        return {
            "total_cost": self.total_cost,
            "total_operations": len(self.operations),
            "breakdown": breakdown,
            "operations": self.operations
        }
    
    def get_cache_savings(self) -> Dict:
        """Calculate savings from caching."""
        total_cached_ops = 0
        estimated_uncached_cost = 0.0
        actual_cost = 0.0
        
        for op in self.operations:
            if "cached" in op:
                actual_cost += op["cost"]
                if op["cached"]:
                    total_cached_ops += 1
                    # Estimate what it would have cost uncached
                    if op["type"] == "embedding":
                        estimated_uncached_cost += op.get("tokens", 0) * 0.0001 / 1000
                    elif op["type"] == "vector_search":
                        estimated_uncached_cost += 0.001
                else:
                    estimated_uncached_cost += op["cost"]
        
        savings = estimated_uncached_cost - actual_cost
        savings_percent = (savings / estimated_uncached_cost * 100) if estimated_uncached_cost > 0 else 0
        
        return {
            "cached_operations": total_cached_ops,
            "estimated_uncached_cost": estimated_uncached_cost,
            "actual_cost": actual_cost,
            "savings": savings,
            "savings_percent": savings_percent
        }
    
    def export_to_json(self, filepath: str):
        """Export cost data to JSON file."""
        data = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cost data exported to {filepath}")
    
    def reset(self):
        """Reset all tracked costs."""
        self.operations = []
        self.total_cost = 0.0
        logger.info("Cost tracker reset")


# Global cost tracker instance
_global_tracker = None

def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
