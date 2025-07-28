#!/usr/bin/env python3
"""
Lightweight CLI for InsightSpike with lazy imports
==================================================

This is the main entry point for the InsightSpike CLI.
All heavy imports are deferred until actually needed.
"""

import os
import sys
from pathlib import Path

# Enable LITE_MODE to prevent heavy imports at package level
os.environ["INSIGHTSPIKE_LITE_MODE"] = "1"

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Create Typer app
app = typer.Typer(
    name="spike",
    help="InsightSpike AI - Discover insights through knowledge synthesis",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

# Global variables for lazy loading
_agent = None
_config = None


def get_or_create_agent(preset: str = "development"):
    """Get or create agent with lazy loading"""
    global _agent, _config
    
    if _agent is None:
        console.print("[dim]Initializing InsightSpike...[/dim]")
        
        # Import only when needed
        from insightspike.config import load_config
        from insightspike.implementations.agents.main_agent import MainAgent
        from insightspike.implementations.datastore.factory import DataStoreFactory
        
        _config = load_config(preset=preset)
        
        # Create datastore
        datastore = DataStoreFactory.create("filesystem", base_path="./data/insightspike")
        
        # Create agent with datastore
        _agent = MainAgent(config=_config, datastore=datastore)
        
        if not _agent.initialize():
            console.print("[red]Failed to initialize agent[/red]")
            raise typer.Exit(1)
            
        # Try to load existing state
        _agent.load_state()
        
    return _agent


@app.command()
def query(
    question: str,
    preset: str = typer.Option(
        "development", help="Config preset: development, experiment, production"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Query the knowledge base and get insights"""
    try:
        console.print(f"[bold blue]💭 Question:[/bold blue] {question}")
        
        # Get agent (lazy loading)
        agent = get_or_create_agent(preset)
        
        # Process with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            
            result = agent.process_question(question, max_cycles=5, verbose=verbose)
            
            progress.update(task, completed=True)
        
        # Display results
        if hasattr(result, 'response'):
            # CycleResult object
            response = result.response
            spike_detected = result.spike_detected
            reasoning_quality = result.reasoning_quality
            retrieved_documents = result.retrieved_documents
        else:
            # Dict format
            response = result.get("response", "No response generated")
            spike_detected = result.get("spike_detected", False)
            reasoning_quality = result.get("reasoning_quality", 0.0)
            retrieved_documents = result.get("documents", [])
        
        console.print(f"\n[bold green]💡 Answer:[/bold green] {response}")
        
        if spike_detected:
            console.print("\n[bold red]🚀 INSIGHT SPIKE DETECTED![/bold red]")
        
        if verbose:
            console.print(f"\n[dim]Quality score: {reasoning_quality:.3f}[/dim]")
            console.print(f"[dim]Retrieved {len(retrieved_documents)} relevant documents[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def embed(
    path: Path,
    preset: str = typer.Option(
        "development", help="Config preset: development, experiment, production"
    ),
):
    """Add documents to the knowledge base"""
    try:
        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(code=1)
        
        # Get agent
        agent = get_or_create_agent(preset)
        
        # Collect files to process
        files_to_process = []
        if path.is_file():
            files_to_process = [path]
        else:
            files_to_process = list(path.glob("**/*.txt")) + list(path.glob("**/*.md"))
        
        if not files_to_process:
            console.print("[yellow]No .txt or .md files found[/yellow]")
            return
        
        # Process files
        console.print(f"[blue]📚 Processing {len(files_to_process)} files...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding documents...", total=len(files_to_process))
            
            for file_path in files_to_process:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    agent.add_knowledge(text=content)
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[yellow]Skipped {file_path}: {e}[/yellow]")
                    progress.update(task, advance=1)
        
        # Save the state
        agent.save_state()
        console.print("[green]✅ Documents added successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def stats():
    """Show agent statistics and insights"""
    try:
        agent = get_or_create_agent()
        stats = agent.get_stats()
        
        # Create stats table
        table = Table(title="Agent Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Initialized", "✅" if stats.get("initialized", False) else "❌")
        table.add_row("Total cycles", str(stats.get("total_cycles", 0)))
        table.add_row("Reasoning history", str(stats.get("reasoning_history_length", 0)))
        table.add_row("Average quality", f"{stats.get('average_quality', 0):.3f}")
        
        # Memory stats
        memory_stats = stats.get("memory_stats", {})
        table.add_row("Total episodes", str(memory_stats.get("total_episodes", 0)))
        table.add_row("Total documents", str(memory_stats.get("total_documents", 0)))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show version information"""
    console.print("[bold]InsightSpike AI[/bold]")
    console.print("Version: 0.8.0")
    console.print("Brain-inspired AI for insight detection")
    console.print("\n[dim]GitHub: https://github.com/miyauchi/InsightSpike-AI[/dim]")


@app.command()
def test():
    """Test command to verify CLI is working"""
    console.print("[green]✅ CLI is working correctly![/green]")
    
    # Test imports
    console.print("\n[dim]Testing lazy imports...[/dim]")
    try:
        from insightspike.config import load_config
        console.print("  ✓ Config module")
    except Exception as e:
        console.print(f"  ✗ Config module: {e}")
        
    try:
        from insightspike.implementations.agents.main_agent import MainAgent
        console.print("  ✓ MainAgent module")
    except Exception as e:
        console.print(f"  ✗ MainAgent module: {e}")


# Aliases for common commands
app.command("q")(query)  # spike q → query
app.command("e")(embed)  # spike e → embed
app.command("l")(embed)  # spike l → embed
app.command("ask")(query)  # spike ask → query
app.command("learn")(embed)  # spike learn → embed


if __name__ == "__main__":
    app()