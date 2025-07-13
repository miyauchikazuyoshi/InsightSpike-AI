#!/usr/bin/env python3
"""
Demo: Query Evolution Tracking and Learning
Shows Phase 3 & 4 features in action
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time
from datetime import datetime
from typing import Dict
import torch
import numpy as np

from src.insightspike.core.agents.main_agent_optimized import MainAgentOptimized
from src.insightspike.core.config import Config
from src.insightspike.core.query_transformation.evolution_tracker import (
    EvolutionPattern, QueryTypeClassifier
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich import print as rprint

console = Console()


def create_performance_dashboard(agent: MainAgentOptimized) -> Layout:
    """Create performance monitoring dashboard"""
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=4)
    )
    
    # Header
    header_text = "[bold cyan]🚀 InsightSpike Performance Dashboard[/bold cyan]"
    layout["header"].update(Panel(header_text))
    
    # Main area - split into columns
    layout["main"].split_row(
        Layout(name="metrics"),
        Layout(name="cache")
    )
    
    # Performance metrics
    perf_summary = agent.get_performance_summary()
    
    metrics_table = Table(show_header=False, padding=1)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Avg Processing Time", f"{perf_summary.get('avg_processing_time', 0):.2f}s")
    metrics_table.add_row("P95 Processing Time", f"{perf_summary.get('p95_processing_time', 0):.2f}s")
    metrics_table.add_row("Cache Hit Rate", f"{perf_summary.get('cache_hit_rate', 0):.1%}")
    metrics_table.add_row("Avg Memory Usage", f"{perf_summary.get('avg_memory_usage_mb', 0):.1f} MB")
    metrics_table.add_row("Total Queries", str(perf_summary.get('total_queries_processed', 0)))
    
    if perf_summary.get('gpu_accelerated'):
        metrics_table.add_row("GPU", "✅ Enabled")
    else:
        metrics_table.add_row("GPU", "❌ Disabled")
    
    layout["metrics"].update(Panel(metrics_table, title="📊 Performance Metrics"))
    
    # Cache status
    cache_table = Table(show_header=False)
    cache_table.add_column("Status", style="magenta")
    cache_table.add_column("Info", style="yellow")
    
    if agent.query_cache:
        cache_size = len(agent.query_cache.cache)
        cache_table.add_row("Cache Size", f"{cache_size}/{agent.query_cache.max_size}")
        
        # Show most accessed queries
        if agent.query_cache.access_counts:
            top_queries = sorted(
                agent.query_cache.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for i, (query, count) in enumerate(top_queries):
                cache_table.add_row(
                    f"Top {i+1}", 
                    f"{query[:30]}... ({count} hits)"
                )
    
    layout["cache"].update(Panel(cache_table, title="💾 Cache Status"))
    
    # Footer - current status
    status = "🟢 Optimized Mode | Learning Enabled | Async Processing"
    layout["footer"].update(Panel(status, title="Status", border_style="green"))
    
    return layout


async def demonstrate_evolution_tracking():
    """Demonstrate evolution tracking and optimization"""
    
    console.print("\n[bold cyan]🧠 Query Evolution Tracking & Optimization Demo[/bold cyan]\n")
    console.print("Phase 3 & 4 Features: Learning, Caching, Performance Optimization\n")
    
    # Initialize optimized agent
    config = Config()
    if hasattr(config, 'llm'):
        config.llm.provider = 'mock'
    
    agent = MainAgentOptimized(
        config=config,
        enable_cache=True,
        enable_learning=True,
        enable_async=True
    )
    
    # Test queries to demonstrate different features
    test_queries = [
        {
            'query': "What is the relationship between entropy and information?",
            'type': 'relational',
            'expected_concepts': ['entropy', 'information', 'thermodynamics']
        },
        {
            'query': "How does quantum tunneling work?",
            'type': 'procedural',
            'expected_concepts': ['quantum', 'barrier', 'probability']
        },
        {
            'query': "What is the relationship between entropy and information?",  # Duplicate for cache test
            'type': 'relational',
            'is_cached': True
        },
        {
            'query': "Compare classical and quantum computing",
            'type': 'comparison',
            'expected_concepts': ['classical', 'quantum', 'bits', 'qubits']
        }
    ]
    
    # Process queries and track evolution
    console.print("[bold]Processing queries with evolution tracking...[/bold]\n")
    
    evolution_patterns = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for i, test_case in enumerate(test_queries):
            task = progress.add_task(
                f"Query {i+1}: {test_case['query'][:50]}...", 
                total=1
            )
            
            start_time = time.time()
            
            # Process query (mock mode for demo)
            if test_case.get('is_cached'):
                # This should hit cache
                result = await agent.process_question_async(test_case['query'])
            else:
                # Mock result for demo
                result = create_mock_result(test_case, agent)
            
            processing_time = time.time() - start_time
            
            progress.update(task, completed=1)
            
            # Display result summary
            if result.get('optimization_metrics', {}).get('cache_hit'):
                console.print(f"  ⚡ Cache hit! Processing time: {processing_time:.3f}s", style="green")
            else:
                console.print(f"  ⏱️  Processing time: {processing_time:.3f}s", style="yellow")
            
            # Track evolution pattern
            if 'evolution_pattern' in result:
                evolution_patterns.append(result['evolution_pattern'])
    
    # Show evolution patterns
    console.print("\n[bold]📈 Evolution Patterns Learned:[/bold]\n")
    
    patterns_table = Table()
    patterns_table.add_column("Query Type", style="cyan")
    patterns_table.add_column("Success Score", style="green")
    patterns_table.add_column("Avg Confidence Gain", style="yellow")
    patterns_table.add_column("Key Concepts", style="magenta")
    
    for pattern in evolution_patterns:
        if pattern:  # Skip cached results
            patterns_table.add_row(
                pattern['initial_query_type'],
                f"{pattern['success_score']:.2f}",
                f"{pattern['avg_confidence_gain']:.2f}",
                ", ".join(pattern['absorbed_concepts'][:3])
            )
    
    console.print(patterns_table)
    
    # Show trajectory analysis
    console.print("\n[bold]🛤️  Query Trajectory Analysis:[/bold]\n")
    
    trajectory_table = Table()
    trajectory_table.add_column("Metric", style="cyan")
    trajectory_table.add_column("Value", style="green")
    trajectory_table.add_column("Interpretation", style="yellow")
    
    # Mock trajectory metrics
    trajectory_metrics = {
        'total_distance': 2.45,
        'trajectory_smoothness': 0.32,
        'direction_changes': 2,
        'confidence_correlation': 0.89
    }
    
    trajectory_table.add_row(
        "Total Distance",
        f"{trajectory_metrics['total_distance']:.2f}",
        "Moderate exploration"
    )
    trajectory_table.add_row(
        "Smoothness",
        f"{trajectory_metrics['trajectory_smoothness']:.2f}",
        "Smooth progression"
    )
    trajectory_table.add_row(
        "Direction Changes",
        str(trajectory_metrics['direction_changes']),
        "Focused exploration"
    )
    trajectory_table.add_row(
        "Confidence Correlation",
        f"{trajectory_metrics['confidence_correlation']:.2f}",
        "Strong positive trend"
    )
    
    console.print(trajectory_table)
    
    # Show optimization suggestions
    console.print("\n[bold]💡 Optimization Suggestions:[/bold]\n")
    
    suggestions = [
        "• Enable multi-hop reasoning for complex relational queries",
        "• Pre-warm cache with frequently accessed concepts",
        "• Use GPU acceleration for embedding generation",
        "• Reduce exploration temperature for well-understood query types"
    ]
    
    for suggestion in suggestions:
        console.print(suggestion, style="green")
    
    # Performance dashboard
    console.print("\n[bold]📊 Performance Dashboard:[/bold]\n")
    
    dashboard = create_performance_dashboard(agent)
    console.print(dashboard)
    
    # Export patterns
    console.print("\n[bold]💾 Exporting learned patterns...[/bold]")
    
    export_path = Path.home() / '.insightspike' / 'exported_patterns.json'
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Mock export
    console.print(f"  ✅ Patterns exported to: {export_path}", style="green")
    
    # Cleanup
    agent.cleanup()
    console.print("\n✨ Demo complete! System has learned from query patterns.\n", style="green")


def create_mock_result(test_case: Dict, agent: MainAgentOptimized) -> Dict:
    """Create mock result for demonstration"""
    
    # Simulate processing
    time.sleep(0.5)
    
    # Create mock evolution pattern
    pattern = EvolutionPattern(
        pattern_id=f"{test_case['type']}_{time.time()}",
        initial_query_type=test_case['type'],
        transformation_path=['initial', 'exploring', 'transforming', 'insight'],
        absorbed_concepts=test_case.get('expected_concepts', []),
        final_insights=[f"Key insight about {test_case['type']} queries"],
        success_score=0.85,
        avg_confidence_gain=0.2,
        num_hops=3,
        branches_used=['general', 'theoretical']
    )
    
    # Track in agent's evolution tracker
    if agent.enable_learning:
        agent.evolution_tracker.pattern_db.save_pattern(pattern)
    
    return {
        'response': f"Processed {test_case['type']} query successfully",
        'evolution_pattern': pattern.to_dict(),
        'optimization_metrics': {
            'processing_time': 0.5,
            'cache_hit': False,
            'gpu_accelerated': agent.device.type == 'cuda',
            'parallel_branches': True
        }
    }


if __name__ == "__main__":
    asyncio.run(demonstrate_evolution_tracking())