"""CLI entrypoints - Updated for new architecture"""
from pathlib import Path
import typer
from rich import print
import pathlib
from typing import Optional

# New imports for refactored structure
from .core.agents.main_agent import MainAgent
from .core.config import get_config
from .loader import load_corpus

app = typer.Typer()

@app.command()
def ask(question: str = typer.Argument(..., help="Ask a question to the AI agent")):
    """Ask a question using the new MainAgent"""
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
        print(f"\n[bold green]Answer:[/bold green] {result.get('response', 'No response generated')}")
        print(f"[dim]Quality: {result.get('reasoning_quality', 0):.3f}, "
              f"Cycles: {result.get('total_cycles', 0)}, "
              f"Spike: {result.get('spike_detected', False)}[/dim]")
        
        if result.get('success', False):
            print("[green]✓ Successfully processed question[/green]")
        else:
            print("[red]✗ Processing failed[/red]")
            
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def load_documents(path: pathlib.Path = typer.Argument(..., help="Path to text file or directory")):
    """Load documents into the agent's memory"""
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
        
    except Exception as e:
        print(f"[red]Error loading documents: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def stats():
    """Show agent statistics"""
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
        
        memory_stats = stats.get('memory_stats', {})
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
        print(f"  Environment: {config.environment}")
        print(f"  LLM Provider: {config.llm.provider}")
        print(f"  Model: {config.llm.model_name}")
        print(f"  Memory max docs: {config.memory.max_retrieved_docs}")
        print(f"  Graph spike thresholds - GED: {config.graph.spike_ged_threshold}, IG: {config.graph.spike_ig_threshold}")
        
    except Exception as e:
        print(f"[red]Error getting config: {e}[/red]")
        raise typer.Exit(code=1)

# Legacy compatibility commands (simplified)
@app.command()
def embed(path: Optional[pathlib.Path] = typer.Option(None, help="Path to text file")):
    """Legacy embed command - use load_documents instead"""
    print("[yellow]Note: 'embed' command is deprecated. Use 'load_documents' instead.[/yellow]")
    
    if path is None:
        raw_dir = Path("data/raw")
        txt_files = sorted(raw_dir.glob("*.txt"))
        if not txt_files:
            print("[red]Error: --path option not specified and no text files in data/raw/[/red]")
            raise typer.Exit(code=1)
        print("[yellow]No path specified, choose from data/raw/ text files:[/yellow]")
        for i, f in enumerate(txt_files):
            print(f"  [{i}] {f.name}")
        idx = typer.prompt("Enter number", type=int)
        if not (0 <= idx < len(txt_files)):
            print("[red]Error: Invalid number[/red]")
            raise typer.Exit(code=1)
        path = txt_files[idx]
    
    # Use the new load_documents command
    load_documents(path)

@app.command()
def query(question: str = typer.Argument(..., help="Question to ask")):
    """Legacy query command - use ask instead"""
    print("[yellow]Note: 'query' command is deprecated. Use 'ask' instead.[/yellow]")
    ask(question)

if __name__ == "__main__":
    app()
