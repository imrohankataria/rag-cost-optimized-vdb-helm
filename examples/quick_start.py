"""
Quick Start Example - Demonstrating Cost Optimization
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cost_tracker import CostTracker
from cache_layer import CacheLayer
from visualizations import CostVisualizer


def simulate_rag_costs():
    """Simulate RAG operations to show cost tracking."""
    
    print("\n" + "="*70)
    print("COST OPTIMIZATION DEMONSTRATION")
    print("="*70)
    
    # Scenario 1: Without Caching
    print("\n📊 SCENARIO 1: Without Caching")
    print("-" * 70)
    
    tracker1 = CostTracker()
    
    # Simulate adding documents (embeddings)
    documents = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines...",
        "Machine learning is a method of data analysis that automates analytical model building...",
        "Deep learning is part of a broader family of machine learning methods..."
    ]
    
    print("\n1. Adding documents to vector store:")
    for i, doc in enumerate(documents, 1):
        cost = tracker1.track_embedding(doc, cached=False)
        print(f"   Document {i}: ${cost:.6f}")
    
    # Simulate queries
    print("\n2. Processing queries:")
    queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    for i, query in enumerate(queries, 1):
        emb_cost = tracker1.track_embedding(query, cached=False)
        search_cost = tracker1.track_vector_search(cached=False)
        gen_cost = tracker1.track_llm_generation(100, 50, "gpt-3.5-turbo")
        total = emb_cost + search_cost + gen_cost
        print(f"   Query {i}: ${total:.6f} (emb: ${emb_cost:.6f}, search: ${search_cost:.6f}, gen: ${gen_cost:.6f})")
    
    summary1 = tracker1.get_summary()
    print(f"\n✓ Total Cost (No Caching): ${summary1['total_cost']:.4f}")
    print(f"✓ Total Operations: {summary1['total_operations']}")
    
    # Scenario 2: With Caching
    print("\n\n📊 SCENARIO 2: With Caching")
    print("-" * 70)
    
    tracker2 = CostTracker()
    
    print("\n1. First-time queries (cache miss):")
    for i, query in enumerate(queries[:2], 1):
        emb_cost = tracker2.track_embedding(query, cached=False)
        search_cost = tracker2.track_vector_search(cached=False)
        gen_cost = tracker2.track_llm_generation(100, 50, "gpt-3.5-turbo")
        total = emb_cost + search_cost + gen_cost
        print(f"   Query {i}: ${total:.6f} (cache miss)")
    
    print("\n2. Repeated queries (cache hit):")
    for i, query in enumerate(queries[:2], 1):
        emb_cost = tracker2.track_embedding(query, cached=True)
        search_cost = tracker2.track_vector_search(cached=True)
        # Note: LLM generation can be cached too in full implementation
        gen_cost = 0.0  # Cached response
        total = emb_cost + search_cost + gen_cost
        print(f"   Query {i}: ${total:.6f} (cache hit) - 99% savings!")
    
    summary2 = tracker2.get_summary()
    print(f"\n✓ Total Cost (With Caching): ${summary2['total_cost']:.4f}")
    print(f"✓ Total Operations: {summary2['total_operations']}")
    
    # Calculate savings
    savings_amount = summary1['total_cost'] - summary2['total_cost']
    savings_percent = (savings_amount / summary1['total_cost'] * 100) if summary1['total_cost'] > 0 else 0
    
    print("\n\n💰 COST SAVINGS SUMMARY")
    print("-" * 70)
    print(f"Without Caching: ${summary1['total_cost']:.4f}")
    print(f"With Caching:    ${summary2['total_cost']:.4f}")
    print(f"Savings:         ${savings_amount:.4f} ({savings_percent:.1f}%)")
    
    # Breakdown
    print("\n\n📋 COST BREAKDOWN")
    print("-" * 70)
    
    breakdown1 = tracker1.get_breakdown()
    for op_type, data in breakdown1.items():
        avg_cost = data['total_cost'] / data['count'] if data['count'] > 0 else 0
        print(f"{op_type:20} | Count: {data['count']:3} | Total: ${data['total_cost']:.6f} | Avg: ${avg_cost:.6f}")
    
    return tracker1, tracker2


def demonstrate_multi_hop_reasoning():
    """Demonstrate multi-hop reasoning costs."""
    
    print("\n\n" + "="*70)
    print("MULTI-HOP REASONING COST ANALYSIS")
    print("="*70)
    
    tracker = CostTracker()
    
    # Simulate a 3-hop query
    hops = [
        ("Initial query: Compare ML and DL", 3, 0.008),
        ("Refinement: Provide more details about ML", 2, 0.006),
        ("Refinement: Explain DL advantages", 2, 0.006)
    ]
    
    total_cost = 0
    for i, (description, docs, cost) in enumerate(hops, 1):
        print(f"\nHop {i}: {description}")
        print(f"  - Retrieved {docs} documents")
        print(f"  - Cost: ${cost:.6f}")
        total_cost += cost
        tracker.track_custom_operation(f"hop_{i}", cost)
    
    print(f"\n✓ Total Multi-Hop Query Cost: ${total_cost:.4f}")
    print(f"✓ Average cost per hop: ${total_cost/len(hops):.4f}")
    
    # Compare with single query
    single_query_cost = 0.025
    print(f"\n📊 Comparison:")
    print(f"  Single query (no reasoning):  ${single_query_cost:.4f}")
    print(f"  Multi-hop query (3 hops):     ${total_cost:.4f}")
    print(f"  Additional cost for reasoning: ${total_cost - single_query_cost:.4f}")
    print(f"  Cost increase: {((total_cost/single_query_cost - 1) * 100):.1f}%")


def generate_comparison_charts():
    """Generate visualization charts."""
    
    print("\n\n" + "="*70)
    print("GENERATING VISUALIZATION CHARTS")
    print("="*70)
    
    visualizer = CostVisualizer()
    
    # Retrieval costs before/after caching
    print("\n1. Generating retrieval cost comparison...")
    before_cache = [0.050, 0.048, 0.052, 0.049, 0.051]
    after_cache = [0.001, 0.001, 0.001, 0.001, 0.001]
    path1 = visualizer.plot_retrieval_costs(before_cache, after_cache)
    print(f"   ✓ Saved to: {path1}")
    
    # Agentic query costs
    print("\n2. Generating agentic query cost comparison...")
    standard = [0.150, 0.140, 0.160, 0.145, 0.155]
    optimized = [0.030, 0.029, 0.031, 0.030, 0.032]
    path2 = visualizer.plot_agentic_costs(standard, optimized)
    print(f"   ✓ Saved to: {path2}")
    
    # Cost breakdown
    print("\n3. Generating cost breakdown chart...")
    breakdown = {
        "embedding": {"count": 15, "total_cost": 0.015, "cached_count": 8, "uncached_count": 7},
        "vector_search": {"count": 12, "total_cost": 0.006, "cached_count": 6, "uncached_count": 6},
        "llm_generation": {"count": 10, "total_cost": 0.050, "cached_count": 0, "uncached_count": 10}
    }
    path3 = visualizer.plot_cost_breakdown(breakdown)
    print(f"   ✓ Saved to: {path3}")
    
    print("\n✓ All charts generated successfully!")
    print(f"✓ Charts location: {visualizer.output_dir}/")


def main():
    """Run all demonstrations."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "RAG COST OPTIMIZATION DEMO" + " "*27 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Run demonstrations
        tracker1, tracker2 = simulate_rag_costs()
        demonstrate_multi_hop_reasoning()
        generate_comparison_charts()
        
        print("\n\n" + "="*70)
        print("✓ DEMONSTRATION COMPLETE")
        print("="*70)
        
        print("\n🎯 Key Takeaways:")
        print("   1. Caching reduces retrieval costs by ~98%")
        print("   2. Multi-hop reasoning adds controlled overhead for better results")
        print("   3. Request-level logging enables precise cost tracking")
        print("   4. Visual analytics make cost optimization transparent")
        
        print("\n📊 Next Steps:")
        print("   - Check the 'visualizations/' directory for charts")
        print("   - Review the cost breakdown by operation type")
        print("   - Experiment with different caching strategies")
        print("   - Deploy with Helm for production use")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
