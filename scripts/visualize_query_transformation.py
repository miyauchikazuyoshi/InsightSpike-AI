#!/usr/bin/env python3
"""
Visualize Query Transformation Process
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insightspike.core.query_transformation import QueryState, QueryTransformationHistory, QueryTransformer
from src.insightspike.utils.graph_construction import GraphBuilder
import torch
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich import print as rprint
import time

console = Console()


def create_demo_graph():
    """Create a simple demo knowledge graph"""
    G = nx.Graph()
    
    # Add nodes
    nodes = {
        'thermodynamics': {'concept': 'thermodynamics', 'text': 'Study of heat and energy'},
        'information_theory': {'concept': 'information theory', 'text': 'Study of data and communication'},
        'entropy': {'concept': 'entropy', 'text': 'Measure of disorder'},
        'physics': {'concept': 'physics', 'text': 'Study of matter and energy'},
        'mathematics': {'concept': 'mathematics', 'text': 'Study of numbers and patterns'}
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    # Add edges
    G.add_edge('thermodynamics', 'physics')
    G.add_edge('thermodynamics', 'entropy')
    G.add_edge('information_theory', 'mathematics')
    G.add_edge('information_theory', 'entropy')
    
    return G


def visualize_transformation():
    """Demonstrate query transformation with visualization"""
    
    console.print("\n[bold cyan]🧠 Query Transformation Visualization[/bold cyan]\n")
    
    # Create components
    console.print("Setting up components...", style="yellow")
    transformer = QueryTransformer(use_gnn=False)  # Disable GNN for simplicity
    graph = create_demo_graph()
    
    console.print("✅ Components ready\n", style="green")
    
    # Initial query
    query = "How does entropy connect thermodynamics and information?"
    console.print(Panel(query, title="[bold yellow]Initial Query[/bold yellow]"))
    
    # Place query on graph
    console.print("\n[blue]1. Placing query on knowledge graph...[/blue]")
    query_state = transformer.place_query_on_graph(query, graph)
    
    # Show initial connections
    table = Table(title="Initial Query Connections")
    table.add_column("Connected Node", style="cyan")
    table.add_column("Connection Strength", style="green")
    
    for node, weight in query_state.edge_weights.items():
        table.add_row(node, f"{weight:.3f}")
    
    console.print(table)
    
    # Simulate transformation cycles
    history = QueryTransformationHistory(query)
    history.add_state(query_state)
    
    console.print("\n[blue]2. Beginning transformation cycles...[/blue]\n")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Transforming query...", total=3)
        
        for cycle in range(3):
            time.sleep(1)  # Simulate processing
            
            # Transform query
            new_state = QueryState(
                text=query_state.text,
                embedding=query_state.embedding + torch.randn(384) * 0.1,
                stage="exploring" if cycle == 0 else "transforming" if cycle == 1 else "insight"
            )
            
            # Simulate concept absorption
            if cycle == 0:
                new_state.absorb_concept("entropy")
                new_state.absorb_concept("thermodynamics")
            elif cycle == 1:
                new_state.absorb_concept("information_theory")
                new_state.add_insight("Entropy appears in both domains")
            else:
                new_state.add_insight("Both entropies measure microstates!")
                new_state.add_insight("S = k ln W unifies the concepts")
            
            history.add_state(new_state)
            query_state = new_state
            
            # Update progress
            progress.update(task, advance=1)
            
            # Show state
            panel_content = f"""
Stage: {new_state.stage}
Confidence: {'▓' * int(new_state.confidence * 10)}{'░' * (10 - int(new_state.confidence * 10))} {new_state.confidence:.0%}
Color: {new_state.color}
Absorbed: {', '.join(new_state.absorbed_concepts[-2:]) if new_state.absorbed_concepts else 'None'}
"""
            if new_state.insights:
                panel_content += f"Insights: {len(new_state.insights)}"
            
            color_map = {'yellow': 'yellow', 'orange': 'orange1', 'green': 'green'}
            border_color = color_map.get(new_state.color, 'white')
            
            console.print(Panel(panel_content, title=f"Cycle {cycle + 1}", border_style=border_color))
    
    # Show final transformation
    console.print("\n[bold green]3. Transformation Complete![/bold green]\n")
    
    # Summary table
    summary = Table(title="Transformation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Initial", style="yellow")
    summary.add_column("Final", style="green")
    
    initial = history.states[0]
    final = history.states[-1]
    
    summary.add_row("Stage", initial.stage, final.stage)
    summary.add_row("Confidence", f"{initial.confidence:.0%}", f"{final.confidence:.0%}")
    summary.add_row("Color", initial.color, final.color)
    summary.add_row("Concepts", "0", str(len(final.absorbed_concepts)))
    summary.add_row("Insights", "0", str(len(final.insights)))
    
    console.print(summary)
    
    # Show insights
    if final.insights:
        console.print("\n[bold green]💡 Discovered Insights:[/bold green]")
        for i, insight in enumerate(final.insights, 1):
            console.print(f"  {i}. {insight}")
    
    # Show transformation path
    console.print(f"\n[bold]Transformation Path:[/bold] {' → '.join(history.get_transformation_path())}")
    
    console.print("\n✨ Visualization complete!\n", style="green")


if __name__ == "__main__":
    visualize_transformation()