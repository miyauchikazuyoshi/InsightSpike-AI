#!/usr/bin/env python3
"""Test script for spike discover command"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.config import get_config
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.cli.commands.discover import InsightDiscovery
from rich.console import Console

console = Console()

def main():
    """Test discover functionality"""
    console.print("[blue]🔍 Testing InsightSpike Discover Feature[/blue]\n")
    
    # Initialize agent
    config = get_config()
    agent = MainAgent(config)
    agent.initialize()
    
    # Create discovery instance
    discovery = InsightDiscovery(agent)
    
    # Load sample documents
    corpus_path = project_root / "data" / "samples"
    console.print(f"[yellow]Loading documents from {corpus_path}...[/yellow]")
    
    files = discovery.analyze_corpus(corpus_path)
    console.print(f"Found {len(files)} files")
    
    loaded = discovery.load_corpus(files)
    console.print(f"[green]✅ Loaded {loaded} documents[/green]\n")
    
    # Add some Q&A to simulate insights
    console.print("[yellow]Simulating some Q&A interactions...[/yellow]")
    
    questions = [
        "What is the relationship between entropy and information?",
        "How do quantum effects appear in biological systems?",
        "Is consciousness related to information integration?"
    ]
    
    for q in questions:
        console.print(f"Q: {q}")
        result = agent.process_question(q, max_cycles=1)
        console.print(f"A: {result.response[:100]}...")
        console.print(f"Quality: {result.reasoning_quality:.3f}\n")
    
    # Discover insights
    console.print("[blue]🔍 Discovering insights...[/blue]\n")
    insights = discovery.discover_insights(min_spike=0.5)
    
    if insights:
        console.print(f"[green bold]⚡ Found {len(insights)} insights![/green bold]\n")
        
        for i, insight in enumerate(insights[:5], 1):
            console.print(f"[bold]Insight #{i}[/bold]")
            console.print(f"Type: {insight.get('type', 'unknown')}")
            console.print(f"Spike value: {insight.get('spike_value', 0):.3f}")
            
            if 'question' in insight:
                console.print(f"Question: {insight['question']}")
            if 'description' in insight:
                console.print(f"Description: {insight['description']}")
                
            console.print(f"Confidence: {insight.get('confidence', 0):.1%}\n")
            
        # Find bridge concepts
        bridges = discovery.find_concept_bridges(insights)
        if bridges:
            console.print("[blue bold]🌉 Bridge Concepts:[/blue bold]")
            for bridge in bridges:
                console.print(f"- {bridge['concept']} (score: {bridge['bridge_score']:.2f})")
    else:
        console.print("[yellow]No insights found. Try adding more documents or lowering the threshold.[/yellow]")
    
    console.print("\n[green]✅ Test completed![/green]")

if __name__ == "__main__":
    main()