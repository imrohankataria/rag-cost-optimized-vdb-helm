"""
Example usage of the RAG Cost-Optimized system.
Demonstrates cost tracking, caching benefits, and multi-hop reasoning.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_system import RAGSystem
from multi_hop_agent import MultiHopAgent
from visualizations import CostVisualizer


def demo_basic_rag():
    """Demonstrate basic RAG with cost tracking."""
    print("=" * 60)
    print("DEMO 1: Basic RAG with Cost Tracking")
    print("=" * 60)
    
    # Initialize system without cache to show costs
    rag = RAGSystem(enable_cache=False)
    
    # Add sample documents
    documents = [
        """Machine learning is a subset of artificial intelligence (AI) that provides 
        systems the ability to automatically learn and improve from experience without 
        being explicitly programmed. Machine learning focuses on the development of 
        computer programs that can access data and use it to learn for themselves.""",
        
        """Deep learning is part of a broader family of machine learning methods based 
        on artificial neural networks with representation learning. Learning can be 
        supervised, semi-supervised or unsupervised. Deep learning architectures such 
        as deep neural networks, deep belief networks, and recurrent neural networks 
        have been applied to fields including computer vision, speech recognition, 
        natural language processing, and more.""",
        
        """Natural language processing (NLP) is a subfield of linguistics, computer 
        science, and artificial intelligence concerned with the interactions between 
        computers and human language, in particular how to program computers to process 
        and analyze large amounts of natural language data. The goal is a computer 
        capable of understanding the contents of documents, including the contextual 
        nuances of the language within them.""",
        
        """Computer vision is an interdisciplinary scientific field that deals with 
        how computers can gain high-level understanding from digital images or videos. 
        From the perspective of engineering, it seeks to understand and automate tasks 
        that the human visual system can do. Computer vision tasks include methods for 
        acquiring, processing, analyzing and understanding digital images."""
    ]
    
    print("\n📚 Adding documents to vector store...")
    add_result = rag.add_documents(documents)
    print(f"✓ Added {add_result['documents_added']} documents")
    print(f"✓ Created {add_result['chunks_created']} chunks")
    print(f"✓ Cost: ${add_result['cost']:.4f}")
    
    # Query without cache
    print("\n🔍 Query 1: Without cache")
    query1 = "What is machine learning?"
    result1 = rag.query(query1, generate_answer=False)
    print(f"Query: {query1}")
    print(f"Retrieved {len(result1['documents'])} documents")
    print(f"Cost: ${result1['cost']:.4f}")
    print(f"Cached: {result1['cached']}")
    
    # Show cost summary
    print("\n💰 Cost Summary:")
    summary = rag.get_cost_summary()
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Total operations: {summary['total_operations']}")
    
    return rag


def demo_caching_benefits():
    """Demonstrate cost savings with caching."""
    print("\n" + "=" * 60)
    print("DEMO 2: Caching Benefits")
    print("=" * 60)
    
    # Initialize with cache enabled
    rag = RAGSystem(enable_cache=True)
    
    # Add documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is a programming language commonly used for web development.",
        "Java is a class-based, object-oriented programming language."
    ]
    
    print("\n📚 Adding documents...")
    rag.add_documents(documents)
    
    query = "Tell me about programming languages"
    
    # First query - no cache
    print("\n🔍 First query (no cache):")
    result1 = rag.query(query, generate_answer=False)
    print(f"Cost: ${result1['cost']:.4f}")
    print(f"Cached: {result1['cached']}")
    
    # Second query - should hit cache
    print("\n🔍 Second query (with cache):")
    result2 = rag.query(query, generate_answer=False)
    print(f"Cost: ${result2['cost']:.4f}")
    print(f"Cached: {result2['cached']}")
    
    # Show cache savings
    print("\n💰 Cache Savings:")
    savings = rag.cost_tracker.get_cache_savings()
    print(f"Cached operations: {savings['cached_operations']}")
    print(f"Estimated uncached cost: ${savings['estimated_uncached_cost']:.4f}")
    print(f"Actual cost: ${savings['actual_cost']:.4f}")
    print(f"Savings: ${savings['savings']:.4f} ({savings['savings_percent']:.1f}%)")
    
    # Cache stats
    print("\n📊 Cache Statistics:")
    cache_stats = rag.cache.get_stats()
    print(f"Hits: {cache_stats['hits']}")
    print(f"Misses: {cache_stats['misses']}")
    print(f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    
    return rag


def demo_multi_hop_reasoning():
    """Demonstrate multi-hop reasoning."""
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Hop Reasoning")
    print("=" * 60)
    
    rag = RAGSystem(enable_cache=True)
    agent = MultiHopAgent(rag, max_hops=3)
    
    # Add comprehensive documents
    documents = [
        "Supervised learning requires labeled training data with input-output pairs.",
        "Unsupervised learning works with unlabeled data to find patterns.",
        "Reinforcement learning involves an agent learning through trial and error.",
        "Image classification is commonly done using supervised learning with CNNs.",
        "Clustering is an unsupervised learning technique for grouping similar data.",
    ]
    
    print("\n📚 Adding documents...")
    rag.add_documents(documents)
    
    # Complex query requiring multiple steps
    query = "Compare supervised and unsupervised learning, then explain which is better for image classification"
    
    print(f"\n🤔 Complex Query: {query}")
    print("\nProcessing with multi-hop reasoning...")
    
    result = agent.query(query, use_cache=True)
    
    print(f"\n✓ Completed in {result['total_hops']} hops")
    print(f"✓ Retrieved {result['total_documents']} total documents")
    print(f"✓ Total cost: ${result['total_cost']:.4f}")
    
    print("\n📋 Reasoning Steps:")
    for step in result['reasoning_steps']:
        print(f"  Hop {step['hop']}: {step['query'][:60]}...")
        print(f"    - Retrieved {step['documents_retrieved']} documents")
        print(f"    - Cost: ${step['cost']:.4f}")
    
    print(f"\n💡 Answer: {result['answer'][:200]}...")
    
    return agent


def demo_visualizations():
    """Generate cost visualization charts."""
    print("\n" + "=" * 60)
    print("DEMO 4: Cost Visualizations")
    print("=" * 60)
    
    visualizer = CostVisualizer()
    
    # Simulate cost data
    print("\n📊 Generating visualization charts...")
    
    # Retrieval costs comparison
    before_cache = [0.050, 0.048, 0.052, 0.049, 0.051, 0.050]
    after_cache = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    
    print("✓ Creating retrieval cost comparison...")
    visualizer.plot_retrieval_costs(before_cache, after_cache)
    
    # Agentic query costs
    standard = [0.150, 0.140, 0.160, 0.145, 0.155, 0.152]
    optimized = [0.030, 0.029, 0.031, 0.030, 0.032, 0.030]
    
    print("✓ Creating agentic query cost comparison...")
    visualizer.plot_agentic_costs(standard, optimized)
    
    # Sample cost breakdown
    breakdown = {
        "embedding": {"count": 10, "total_cost": 0.010, "cached_count": 5, "uncached_count": 5},
        "vector_search": {"count": 8, "total_cost": 0.004, "cached_count": 4, "uncached_count": 4},
        "llm_generation": {"count": 5, "total_cost": 0.025, "cached_count": 0, "uncached_count": 5}
    }
    
    print("✓ Creating cost breakdown chart...")
    visualizer.plot_cost_breakdown(breakdown)
    
    print(f"\n✓ All visualizations saved to: {visualizer.output_dir}")
    print("\nVisualization files:")
    print("  - retrieval_costs.png")
    print("  - agentic_costs.png")
    print("  - cost_breakdown.png")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("RAG COST-OPTIMIZED SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Run demos
        demo_basic_rag()
        demo_caching_benefits()
        demo_multi_hop_reasoning()
        demo_visualizations()
        
        print("\n" + "=" * 60)
        print("✓ All demos completed successfully!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Caching reduces retrieval costs by ~98%")
        print("2. Multi-hop reasoning enables complex queries")
        print("3. Request-level logging tracks all costs")
        print("4. Visualizations show cost optimization impact")
        print("\nCheck the 'visualizations' directory for cost charts!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
