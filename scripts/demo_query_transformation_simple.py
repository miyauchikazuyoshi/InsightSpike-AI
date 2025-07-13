#!/usr/bin/env python3
"""
Simple Demo: Query Transformation in InsightSpike
Shows basic query evolution without GNN complications
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform
from src.insightspike.core.config import Config
from rich.console import Console
from rich import print as rprint

console = Console()


def demonstrate_simple_transformation():
    """Run a simple demonstration without GNN"""
    
    console.print("\n[bold cyan]🧠 InsightSpike Query Transformation Demo (Simple)[/bold cyan]\n")
    
    # Initialize agent WITHOUT GNN
    console.print("Initializing InsightSpike (GNN disabled for testing)...", style="yellow")
    
    config = Config()
    # Force simple LLM provider
    if hasattr(config, 'llm'):
        config.llm.provider = 'mock'  # Use mock for demo
    agent = MainAgentWithQueryTransform(config, enable_query_transformation=True)
    
    # Disable GNN for simpler testing
    if hasattr(agent.query_transformer, 'use_gnn'):
        agent.query_transformer.use_gnn = False
        console.print("✅ GNN disabled for simple demonstration\n", style="green")
    
    # Initialize the agent
    if not agent.initialize():
        console.print("❌ Failed to initialize agent", style="red")
        return
    
    # Add some knowledge
    console.print("📚 Adding knowledge...", style="blue")
    knowledge = [
        "Entropy measures disorder in systems.",
        "Information theory uses entropy to measure uncertainty.",
        "Both types of entropy use logarithms in their formulas."
    ]
    
    for k in knowledge:
        agent.add_episode_with_graph_update(k)
        console.print(f"  ✓ {k}", style="dim")
    
    console.print("\n✅ Ready!\n", style="green")
    
    # Process a question
    question = "What is entropy?"
    
    console.print(f"❓ Question: [yellow]{question}[/yellow]\n")
    
    # Create mock result for demo
    console.print("Using mock result for demonstration...", style="yellow")
    
    # Mock a simple query transformation result
    from src.insightspike.core.query_transformation import QueryState, QueryTransformationHistory
    import torch
    
    # Create transformation history
    history = QueryTransformationHistory(question)
    
    # Initial state
    state1 = QueryState(
        text=question,
        embedding=torch.randn(384),
        stage="initial",
        confidence=0.1
    )
    history.add_state(state1)
    
    # Exploring state
    state2 = QueryState(
        text=question,
        embedding=torch.randn(384),
        stage="exploring",
        confidence=0.4,
        absorbed_concepts=["disorder", "uncertainty"]
    )
    state2.add_insight("Entropy measures both disorder and uncertainty")
    history.add_state(state2)
    
    # Final state
    state3 = QueryState(
        text=question,
        embedding=torch.randn(384),
        stage="insight",
        confidence=0.8,
        absorbed_concepts=["disorder", "uncertainty", "logarithms"]
    )
    state3.add_insight("Entropy quantifies unpredictability using logarithmic scales")
    history.add_state(state3)
    
    result = {
        "response": "Entropy is a measure of disorder or uncertainty in a system, quantified using logarithms.",
        "query_evolution": {
            "initial": question,
            "final_state": state3.to_dict(),
            "insights_discovered": history.get_total_insights()
        }
    }
    
    # Show results
    if "response" in result:
        console.print(f"📝 Answer: {result['response']}\n", style="green")
    
    if "query_evolution" in result:
        evolution = result["query_evolution"]
        console.print("[bold]Query Evolution:[/bold]")
        console.print(f"  Initial query: {evolution.get('initial', question)}")
        
        final_state = evolution.get('final_state', {})
        if final_state:
            console.print(f"  Final stage: {final_state.get('stage', 'unknown')}")
            console.print(f"  Confidence: {final_state.get('confidence', 0):.1%}")
            
            if final_state.get('absorbed_concepts'):
                console.print(f"  Concepts absorbed: {', '.join(final_state['absorbed_concepts'])}")
            
            if final_state.get('insights'):
                console.print("\n[bold green]💡 Insights discovered:[/bold green]")
                for insight in final_state['insights']:
                    console.print(f"    • {insight}")
    
    console.print("\n✨ Demo complete!\n", style="green")


if __name__ == "__main__":
    demonstrate_simple_transformation()