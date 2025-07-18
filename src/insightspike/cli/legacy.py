"""CLI entrypoints - Cleaned version with only essential commands"""
import json
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print

from ..config import get_config
from ..detection.insight_registry import InsightFactRegistry

# New imports for refactored structure
from ..implementations.agents.main_agent import MainAgent
from ..processing.loader import load_corpus

# Import dependency management commands
# Temporarily disabled due to import issues
# from .commands.deps import deps_app
deps_app = None

app = typer.Typer(help="InsightSpike Legacy CLI - Use 'spike' for new features")

# Add dependency management commands as a subcommand group
if deps_app:
    app.add_typer(deps_app, name="deps")


@app.command("legacy-ask")
def ask(question: str = typer.Argument(..., help="Ask a question to the AI agent")):
    """[DEPRECATED] Use 'spike ask' instead. Legacy ask command."""
    print(
        "[yellow]⚠️  This command is deprecated. Please use 'spike ask' instead.[/yellow]\n"
    )
    try:
        print(f"[bold blue]Question:[/bold blue] {question}")
        print("[yellow]Initializing AI agent...[/yellow]")

        # Create and initialize agent
        agent = MainAgent()
        if not agent.initialize():
            print("[red]Failed to initialize agent[/red]")
            raise typer.Exit(code=1)

        # Process the question
        print("[yellow]Processing question...[/yellow]")
        result = agent.process_question(question, max_cycles=5, verbose=True)

        # Display results
        print(
            f"\n[bold green]Answer:[/bold green] {result.get('response', 'No response generated')}"
        )
        print(
            f"[dim]Quality: {result.get('reasoning_quality', 0):.3f}, "
            f"Cycles: {result.get('total_cycles', 0)}, "
            f"Spike: {result.get('spike_detected', False)}[/dim]"
        )

        if result.get("success", False):
            print("[green]✓ Successfully processed question[/green]")
        else:
            print("[red]✗ Processing failed[/red]")

    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def load_documents(
    path: pathlib.Path = typer.Argument(..., help="Path to text file or directory")
):
    """Load documents into the agent's memory (without graph update)"""
    print(
        "[yellow]⚠️  Note: This command does NOT update the graph. Use 'spike learn' for full functionality.[/yellow]\n"
    )
    try:
        print(f"[yellow]Loading documents from: {path}[/yellow]")

        # Create agent
        agent = MainAgent()
        if not agent.initialize():
            print("[red]Failed to initialize agent[/red]")
            raise typer.Exit(code=1)

        # Load documents
        if path.is_file():
            docs = load_corpus(path)
        elif path.is_dir():
            docs = []
            for txt_file in path.glob("*.txt"):
                docs.extend(load_corpus(txt_file))
        else:
            print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(code=1)

        # Add to memory
        added = 0
        for doc in docs:
            if agent.add_document(doc):
                added += 1

        print(f"[green]Successfully loaded {added}/{len(docs)} documents[/green]")
        print(
            "[yellow]Remember: Documents loaded but graph NOT updated. Data NOT saved.[/yellow]"
        )

    except Exception as e:
        print(f"[red]Error loading documents: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("legacy-stats")
def stats():
    """[DEPRECATED] Use 'spike stats' instead. Show agent statistics"""
    print(
        "[yellow]⚠️  This command is deprecated. Please use 'spike stats' instead.[/yellow]\n"
    )
    try:
        agent = MainAgent()
        if not agent.initialize():
            print("[red]Failed to initialize agent[/red]")
            raise typer.Exit(code=1)

        stats = agent.get_stats()

        print("[bold blue]Agent Statistics:[/bold blue]")
        print(f"  Initialized: {stats.get('initialized', False)}")
        print(f"  Total cycles: {stats.get('total_cycles', 0)}")
        print(f"  Reasoning history: {stats.get('reasoning_history_length', 0)}")
        print(f"  Average quality: {stats.get('average_quality', 0):.3f}")

        memory_stats = stats.get("memory_stats", {})
        print(f"\n[bold blue]Memory Statistics:[/bold blue]")
        print(f"  Total episodes: {memory_stats.get('total_episodes', 0)}")
        print(f"  Total documents: {memory_stats.get('total_documents', 0)}")
        print(f"  Index type: {memory_stats.get('index_type', 'Unknown')}")

    except Exception as e:
        print(f"[red]Error getting stats: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def config_info():
    """Show current configuration"""
    try:
        config = get_config()

        print("[bold blue]Current Configuration:[/bold blue]")
        # Handle both Pydantic and legacy config
        if hasattr(config, "environment"):
            print(f"  Environment: {config.environment}")
        if hasattr(config, "llm"):
            print(f"  LLM Provider: {config.llm.provider}")
            print(
                f"  Model: {getattr(config.llm, 'model', config.llm.model_name if hasattr(config.llm, 'model_name') else 'N/A')}"
            )
        if hasattr(config, "memory"):
            print(f"  Memory max docs: {config.memory.max_retrieved_docs}")
        if hasattr(config, "graph"):
            print(
                f"  Graph spike thresholds - GED: {config.graph.spike_ged_threshold}, IG: {config.graph.spike_ig_threshold}"
            )

    except Exception as e:
        print(f"[red]Error getting config: {e}[/red]")
        raise typer.Exit(code=1)


@app.callback()
def main_callback():
    """
    InsightSpike Legacy CLI

    ⚠️  This is the legacy CLI. For new features, use 'spike' instead:

    Examples:
      spike query "Your question"    # Query the knowledge base
      spike embed documents.txt      # Embed documents
      spike chat                     # Interactive mode
      spike config show              # View configuration
    """
    pass


def main():
    """Main CLI entry point for Poetry scripts"""
    app()


if __name__ == "__main__":
    main()
