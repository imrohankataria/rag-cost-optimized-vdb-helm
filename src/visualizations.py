"""
Visualization module for cost analytics and comparisons.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)


class CostVisualizer:
    """Generate visualizations for cost analysis."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_retrieval_costs(self,
                            before_cache: List[float],
                            after_cache: List[float],
                            save_path: Optional[str] = None) -> str:
        """
        Plot retrieval cost comparison before and after caching.
        
        Args:
            before_cache: List of costs before caching
            after_cache: List of costs after caching
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "retrieval_costs.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(before_cache))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_cache, width, label='Before Cache', color='#ff6b6b')
        bars2 = ax.bar(x + width/2, after_cache, width, label='After Cache', color='#4ecdc4')
        
        ax.set_xlabel('Query Number', fontsize=12)
        ax.set_ylabel('Cost ($)', fontsize=12)
        ax.set_title('Retrieval Cost: Before vs After Caching', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{i+1}' for i in range(len(before_cache))])
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Calculate and display savings
        avg_before = np.mean(before_cache)
        avg_after = np.mean(after_cache)
        savings_pct = ((avg_before - avg_after) / avg_before * 100)
        
        ax.text(0.02, 0.98, f'Average Savings: {savings_pct:.1f}%',
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved retrieval cost plot to {save_path}")
        return save_path
    
    def plot_agentic_costs(self,
                          standard: List[float],
                          optimized: List[float],
                          save_path: Optional[str] = None) -> str:
        """
        Plot agentic query cost comparison.
        
        Args:
            standard: List of standard query costs
            optimized: List of optimized query costs
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "agentic_costs.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(standard))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, standard, width, label='Standard', color='#e74c3c')
        bars2 = ax.bar(x + width/2, optimized, width, label='Optimized', color='#2ecc71')
        
        ax.set_xlabel('Query Number', fontsize=12)
        ax.set_ylabel('Cost ($)', fontsize=12)
        ax.set_title('Agentic Query Cost: Standard vs Optimized', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{i+1}' for i in range(len(standard))])
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Calculate and display savings
        avg_standard = np.mean(standard)
        avg_optimized = np.mean(optimized)
        savings_pct = ((avg_standard - avg_optimized) / avg_standard * 100)
        
        ax.text(0.02, 0.98, f'Average Savings: {savings_pct:.1f}%',
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved agentic cost plot to {save_path}")
        return save_path
    
    def plot_cost_breakdown(self,
                           breakdown: Dict,
                           save_path: Optional[str] = None) -> str:
        """
        Plot cost breakdown by operation type.
        
        Args:
            breakdown: Cost breakdown dictionary
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "cost_breakdown.png")
        
        operations = list(breakdown.keys())
        costs = [breakdown[op]["total_cost"] for op in operations]
        counts = [breakdown[op]["count"] for op in operations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost breakdown pie chart
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce']
        ax1.pie(costs, labels=operations, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Cost Breakdown by Operation Type', fontsize=14, fontweight='bold')
        
        # Operation count bar chart
        bars = ax2.bar(operations, counts, color=colors[:len(operations)])
        ax2.set_xlabel('Operation Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Operation Counts', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cost breakdown plot to {save_path}")
        return save_path
    
    def plot_cache_impact(self,
                         cache_stats: Dict,
                         save_path: Optional[str] = None) -> str:
        """
        Plot cache hit/miss statistics.
        
        Args:
            cache_stats: Cache statistics dictionary
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "cache_impact.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cache hit/miss pie chart
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        
        if hits + misses > 0:
            ax1.pie([hits, misses], 
                   labels=['Cache Hits', 'Cache Misses'],
                   autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'],
                   startangle=90)
            ax1.set_title('Cache Hit Rate', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No cache data', ha='center', va='center')
            ax1.set_title('Cache Hit Rate', fontsize=14, fontweight='bold')
        
        # Cost savings bar chart
        savings_data = cache_stats.get("savings", {})
        categories = ['Uncached\nCost', 'Actual\nCost', 'Savings']
        values = [
            savings_data.get("estimated_uncached_cost", 0),
            savings_data.get("actual_cost", 0),
            savings_data.get("savings", 0)
        ]
        colors_bar = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars = ax2.bar(categories, values, color=colors_bar)
        ax2.set_ylabel('Cost ($)', fontsize=12)
        ax2.set_title('Cost Savings from Caching', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'${height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cache impact plot to {save_path}")
        return save_path
    
    def generate_all_plots(self, cost_summary: Dict) -> List[str]:
        """
        Generate all visualization plots from cost summary.
        
        Args:
            cost_summary: Complete cost summary dictionary
            
        Returns:
            List of paths to saved plots
        """
        saved_plots = []
        
        # Generate cost breakdown
        if "breakdown" in cost_summary:
            plot_path = self.plot_cost_breakdown(cost_summary["breakdown"])
            saved_plots.append(plot_path)
        
        # Generate cache impact plot
        if "cache_stats" in cost_summary:
            cache_stats_with_savings = {
                **cost_summary["cache_stats"],
                "savings": cost_summary.get("cache_savings", {})
            }
            plot_path = self.plot_cache_impact(cache_stats_with_savings)
            saved_plots.append(plot_path)
        
        logger.info(f"Generated {len(saved_plots)} visualization plots")
        return saved_plots


if __name__ == "__main__":
    # Example usage
    visualizer = CostVisualizer()
    
    # Example data
    before_cache = [0.05, 0.048, 0.052, 0.049, 0.051]
    after_cache = [0.001, 0.001, 0.001, 0.001, 0.001]
    
    standard = [0.15, 0.14, 0.16, 0.145, 0.155]
    optimized = [0.03, 0.029, 0.031, 0.030, 0.032]
    
    visualizer.plot_retrieval_costs(before_cache, after_cache)
    visualizer.plot_agentic_costs(standard, optimized)
    
    print("Generated example visualizations")
