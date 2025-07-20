#!/usr/bin/env python3
"""Minimal test for discover functionality without full initialization"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from insightspike.cli.commands.discover import InsightDiscovery
from rich.console import Console

console = Console()

# Mock agent for testing
class MockAgent:
    def __init__(self):
        self.reasoning_history = [
            {"question": "What is entropy?", "quality": 0.85},
            {"question": "How does quantum biology work?", "quality": 0.92},
        ]
    
    def get_memory_graph_state(self):
        return {
            "graph": {"num_nodes": 25, "num_edges": 48},
            "memory": {"total_episodes": 15}
        }
    
    def add_knowledge(self, text):
        pass

def main():
    console.print("[blue]🔍 Testing Discover Output Format[/blue]\n")
    
    # Create mock discovery
    mock_agent = MockAgent()
    discovery = InsightDiscovery(mock_agent)
    
    # Test insight discovery
    insights = discovery.discover_insights(min_spike=0.5)
    
    # Display results
    if insights:
        console.print(f"[green bold]⚡ Discovered {len(insights)} insights[/green bold]\n")
        
        for i, insight in enumerate(insights[:3], 1):
            spike_value = insight.get('spike_value', 0)
            
            # Test output formatting
            from rich.panel import Panel
            
            if spike_value >= 0.9:
                color = "red"
                emoji = "🔥"
            else:
                color = "blue"
                emoji = "💡"
            
            panel = Panel(
                f"Question: {insight.get('question', 'N/A')}\nConfidence: {insight.get('confidence', 0):.0%}",
                title=f"{emoji} Insight #{i} [Spike: {spike_value:.2f}]",
                border_style=color
            )
            console.print(panel)
        
        # Test bridge concepts
        bridges = discovery.find_concept_bridges(insights)
        if bridges:
            console.print("\n[blue bold]🌉 Bridge Concepts:[/blue bold]")
            from rich.table import Table
            
            table = Table(show_header=True)
            table.add_column("Concept")
            table.add_column("Score")
            
            for bridge in bridges[:3]:
                table.add_row(bridge['concept'], f"{bridge['bridge_score']:.2f}")
            
            console.print(table)
    
    console.print("\n[green]✅ Output format test completed![/green]")

if __name__ == "__main__":
    main()