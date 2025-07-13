#!/usr/bin/env python3
"""
Demo: Advanced Query Transformation with Multi-hop and Branching
Shows Phase 2 features in action
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insightspike.core.agents.main_agent_advanced import MainAgentAdvanced
from src.insightspike.core.config import Config
from src.insightspike.core.query_transformation import QueryState, QueryTransformationHistory
from src.insightspike.core.query_transformation.enhanced_query_transformer import QueryBranch
import torch
import logging

# Suppress warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.columns import Columns
from rich import print as rprint

console = Console()


def visualize_branches(branches):
    """Visualize exploration branches as a tree"""
    
    tree = Tree("🌳 Query Exploration Branches")
    
    for branch in branches:
        branch_node = tree.add(
            f"[bold {branch.exploration_direction}]{branch.branch_id}[/] "
            f"({branch.exploration_direction})"
        )
        
        # Add branch details
        branch_node.add(f"Priority: {branch.priority:.2f}")
        branch_node.add(f"Confidence: {branch.current_state.confidence:.0%}")
        
        # Add insights
        if branch.current_state.insights:
            insights_node = branch_node.add("💡 Insights")
            for insight in branch.current_state.insights:
                insights_node.add(f"[green]{insight}[/green]")
        
        # Add absorbed concepts
        if branch.current_state.absorbed_concepts:
            concepts_node = branch_node.add("🧩 Absorbed Concepts")
            for concept in branch.current_state.absorbed_concepts[-3:]:
                concepts_node.add(f"[blue]{concept}[/blue]")
    
    return tree


def visualize_reasoning_paths(paths):
    """Visualize multi-hop reasoning paths"""
    
    table = Table(title="🔄 Multi-hop Reasoning Paths")
    table.add_column("Path #", style="cyan", width=8)
    table.add_column("Reasoning Path", style="magenta")
    table.add_column("Hops", style="green", width=6)
    
    for i, path in enumerate(paths):
        path_str = " → ".join(path)
        table.add_row(str(i + 1), path_str, str(len(path)))
    
    return table


def demonstrate_advanced_transformation():
    """Run advanced transformation demonstration"""
    
    console.print("\n[bold cyan]🧠 Advanced Query Transformation Demo (Phase 2)[/bold cyan]\n")
    console.print("Features: Multi-hop reasoning, Adaptive exploration, Query branching\n")
    
    # Create mock result for demonstration
    console.print("Creating mock advanced transformation result...\n", style="yellow")
    
    # Initial query
    question = "How do quantum mechanics and consciousness relate through information theory?"
    console.print(Panel(question, title="❓ Complex Query", border_style="yellow"))
    
    # Mock transformation history
    history = QueryTransformationHistory(question)
    
    # Initial state
    state1 = QueryState(
        text=question,
        embedding=torch.randn(384),
        stage="initial",
        confidence=0.1
    )
    history.add_state(state1)
    
    # Create branches for different exploration directions
    console.print("\n[bold]Creating exploration branches...[/bold]\n")
    
    # Branch 1: Physics perspective
    physics_branch = QueryBranch(
        branch_id="branch_0_physics",
        parent_state=state1,
        current_state=QueryState(
            text=question,
            embedding=torch.randn(384),
            stage="exploring",
            confidence=0.4,
            absorbed_concepts=["quantum_mechanics", "wave_function", "measurement"],
            insights=["Quantum measurement requires conscious observation"]
        ),
        exploration_direction="physics",
        priority=1.0
    )
    
    # Branch 2: Neuroscience perspective  
    neuro_branch = QueryBranch(
        branch_id="branch_1_neuroscience",
        parent_state=state1,
        current_state=QueryState(
            text=question,
            embedding=torch.randn(384),
            stage="exploring",
            confidence=0.5,
            absorbed_concepts=["consciousness", "neural_networks", "emergence"],
            insights=["Consciousness emerges from information integration"]
        ),
        exploration_direction="neuroscience",
        priority=0.5
    )
    
    # Branch 3: Information theory perspective
    info_branch = QueryBranch(
        branch_id="branch_2_information",
        parent_state=state1,
        current_state=QueryState(
            text=question,
            embedding=torch.randn(384),
            stage="insight",
            confidence=0.8,
            absorbed_concepts=["information", "entropy", "computation"],
            insights=["Information is fundamental to both quantum mechanics and consciousness"]
        ),
        exploration_direction="information_theory",
        priority=0.33
    )
    
    branches = [physics_branch, neuro_branch, info_branch]
    
    # Display branches
    branch_tree = visualize_branches(branches)
    console.print(branch_tree)
    
    # Mock reasoning paths
    console.print("\n[bold]Multi-hop reasoning discovered:[/bold]\n")
    
    reasoning_paths = [
        ["quantum_mechanics", "measurement", "observer", "consciousness"],
        ["information", "entropy", "thermodynamics", "quantum_mechanics"],
        ["consciousness", "information_integration", "quantum_coherence"],
        ["neural_networks", "quantum_computation", "information_theory"]
    ]
    
    paths_table = visualize_reasoning_paths(reasoning_paths)
    console.print(paths_table)
    
    # Final synthesized state
    final_state = QueryState(
        text=question,
        embedding=torch.randn(384),
        stage="insight",
        confidence=0.9,
        absorbed_concepts=[
            "quantum_mechanics", "consciousness", "information_theory",
            "measurement", "observer_effect", "integrated_information"
        ]
    )
    
    # Synthesized insights
    console.print("\n[bold green]💡 Synthesized Insights:[/bold green]\n")
    
    insights = [
        "Information serves as the bridge between quantum mechanics and consciousness",
        "Both quantum measurement and conscious experience involve information collapse",
        "Integrated Information Theory (IIT) provides a mathematical framework",
        "Observer effect in QM parallels the role of consciousness in information processing"
    ]
    
    for i, insight in enumerate(insights, 1):
        console.print(f"  {i}. {insight}")
    
    # Advanced metrics
    console.print("\n[bold]Advanced Metrics:[/bold]\n")
    
    metrics_table = Table(show_header=False)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Multi-hop Coverage", "4 paths")
    metrics_table.add_row("Branch Diversity", "3 perspectives")
    metrics_table.add_row("Convergence Rate", "87%")
    metrics_table.add_row("Insight Density", "1.33 per branch")
    metrics_table.add_row("Exploration Strategy", "Adaptive (temp=0.51)")
    
    console.print(metrics_table)
    
    # Exploration summary
    console.print("\n[bold]Exploration Summary:[/bold]")
    console.print("Explored 3 directions: physics, neuroscience, information_theory.")
    console.print("Found insights in: information_theory, synthesized across all branches.\n")
    
    # Final answer
    final_answer = (
        "Quantum mechanics and consciousness are connected through information theory. "
        "Both involve the collapse of information states - quantum measurement collapses "
        "wave functions while conscious observation collapses probability distributions. "
        "Integrated Information Theory provides a mathematical framework showing how "
        "consciousness emerges from information integration, paralleling quantum coherence."
    )
    
    console.print(Panel(final_answer, title="[bold green]📝 Synthesized Answer[/bold green]", 
                       border_style="green"))
    
    # Show convergence visualization
    console.print("\n[bold]Query State Evolution:[/bold]\n")
    
    evolution_table = Table()
    evolution_table.add_column("Stage", style="cyan")
    evolution_table.add_column("Main", style="yellow") 
    evolution_table.add_column("Physics", style="blue")
    evolution_table.add_column("Neuro", style="magenta")
    evolution_table.add_column("Info", style="green")
    
    evolution_table.add_row(
        "Initial",
        "◯ 10%", "◯ 10%", "◯ 10%", "◯ 10%"
    )
    evolution_table.add_row(
        "Exploring", 
        "◐ 40%", "◐ 40%", "◐ 50%", "◐ 60%"
    )
    evolution_table.add_row(
        "Insight",
        "◉ 90%", "◐ 60%", "◕ 70%", "◉ 80%"
    )
    
    console.print(evolution_table)
    
    console.print("\n✨ Advanced transformation complete!\n", style="green")


if __name__ == "__main__":
    demonstrate_advanced_transformation()