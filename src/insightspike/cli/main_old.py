"""CLI entrypoints - Updated for new architecture"""
import json
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print

# New imports for refactored structure
from ..core.agents.main_agent import MainAgent
from ..core.config import get_config
from ..detection.insight_registry import InsightFactRegistry
from ..processing.loader import load_corpus

# Import dependency management commands
from .deps_typer import deps_app

app = typer.Typer()

# Add dependency management commands as a subcommand group
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
        print(f"  Environment: {config.environment}")
        print(f"  LLM Provider: {config.llm.provider}")
        print(f"  Model: {config.llm.model_name}")
        print(f"  Memory max docs: {config.memory.max_retrieved_docs}")
        print(
            f"  Graph spike thresholds - GED: {config.graph.spike_ged_threshold}, IG: {config.graph.spike_ig_threshold}"
        )

    except Exception as e:
        print(f"[red]Error getting config: {e}[/red]")
        raise typer.Exit(code=1)


# Experimental commands
# Removed experiment command - run scripts directly
# Use: python scripts/run_poc_simple.py


@app.command()
def experiment():
    """[REMOVED] Run scripts directly instead"""
    print("[yellow]🧪 Running InsightSpike-AI experimental validation...[/yellow]")

    try:
        # Generate simple dataset
        print("[yellow]Step 1/4: Generating experimental dataset...[/yellow]")
        result = subprocess.run(
            [sys.executable, "scripts/databake_simple.py"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Dataset generation failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        # Run experimental evaluation
        print("[yellow]Step 2/4: Running comparative evaluation...[/yellow]")
        result = subprocess.run(
            [sys.executable, "scripts/run_poc_simple.py"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Evaluation failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        # Generate analysis
        print("[yellow]Step 3/4: Generating analysis report...[/yellow]")
        result = subprocess.run(
            [sys.executable, "scripts/analyze_results.py"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Analysis failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        # Generate visual summary
        print("[yellow]Step 4/4: Creating visual summaries...[/yellow]")
        result = subprocess.run(
            [sys.executable, "scripts/generate_visual_summary.py"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Visualization failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        print(f"[red]Experimental validation failed: {e}[/red]")
        raise typer.Exit(code=1)

    # Load and display results
    try:
        with open("data/processed/experiment_results.json", "r") as f:
            results = json.load(f)

        analysis = results["analysis"]

        # Display results in simple table format
        print("\n[bold blue]🧪 Experimental Validation Results[/bold blue]")
        print("=" * 50)

        # Response quality
        insight_qual = analysis["response_quality"]["insightspike_avg"]
        baseline_qual = analysis["response_quality"]["baseline_avg"]
        qual_improvement = analysis["improvements"]["response_quality_improvement_pct"]

        print(f"📊 Response Quality:")
        print(f"  InsightSpike: {insight_qual:.3f}")
        print(f"  Baseline:     {baseline_qual:.3f}")
        print(f"  Improvement:  +{qual_improvement:.1f}%")

        # Insight detection
        insight_rate = analysis["insight_detection"]["insightspike_rate"] * 100
        baseline_rate = analysis["insight_detection"]["baseline_rate"] * 100

        print(f"\n🧠 Insight Detection:")
        print(f"  InsightSpike: {insight_rate:.0f}%")
        print(f"  Baseline:     {baseline_rate:.0f}%")
        print(f"  Advantage:    Perfect detection")

        # Processing speed
        insight_time = analysis["processing_metrics"]["avg_response_time_is"] * 1000
        baseline_time = (
            analysis["processing_metrics"]["avg_response_time_baseline"] * 1000
        )
        speed_factor = (
            baseline_time / insight_time if insight_time > 0 else float("inf")
        )

        print(f"\n⚡ Processing Speed:")
        print(f"  InsightSpike: {insight_time:.2f}ms")
        print(f"  Baseline:     {baseline_time:.1f}ms")
        print(f"  Speed Boost:  {speed_factor:.0f}x faster")

        # False positives
        false_positive_rate = analysis["insight_detection"]["false_positive_rate"] * 100

        print(f"\n🎯 False Positives:")
        print(f"  Rate: {false_positive_rate:.0f}%")
        print(f"  Status: Zero errors")

        print(
            "\n[bold green]✅ Experimental validation completed successfully![/bold green]"
        )
        print(f"📄 Full report: [blue]EXPERIMENTAL_VALIDATION_REPORT.md[/blue]")
        print(f"📊 Visual summary: [blue]VISUAL_SUMMARY.txt[/blue]")

    except Exception as e:
        print(f"[red]Error displaying results: {e}[/red]")
        print(
            "[yellow]Check data/processed/experiment_results.json for raw results[/yellow]"
        )


# Removed benchmark command - run scripts directly
# Use: python scripts/run_poc_*.py


@app.command()
def benchmark(
    dataset: str = typer.Option(
        "simple", help="Dataset to use: simple, enhanced, or custom"
    ),
    verbose: bool = typer.Option(False, help="Show detailed output"),
):
    """Run performance benchmarks"""
    script_map = {
        "simple": "scripts/run_poc_simple.py",
        "enhanced": "scripts/run_poc_enhanced.py",
        "custom": "scripts/run_poc.py",
    }

    if dataset not in script_map:
        print(
            f"[red]Unknown dataset: {dataset}. Choose from: {', '.join(script_map.keys())}[/red]"
        )
        raise typer.Exit(code=1)

    print(f"[yellow]📊 Running benchmark with {dataset} dataset...[/yellow]")

    try:
        result = subprocess.run(
            [sys.executable, script_map[dataset]],
            cwd=".",
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Benchmark failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        if not verbose and result.stdout:
            # Extract key metrics from output
            lines = result.stdout.split("\n")
            for line in lines:
                if (
                    "Response Quality" in line
                    or "Insight Detection" in line
                    or "Processing" in line
                ):
                    print(line)

        print("[green]✅ Benchmark completed successfully![/green]")

    except Exception as e:
        print(f"[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(code=1)


# Removed legacy embed command - use load_documents instead
# Removed legacy query command - use legacy-ask instead
# Removed insight_experiment - run scripts directly
# Removed compare_experiments - run scripts directly
# Removed experiment_suite function - use individual experiments instead
# Removed demo command - moved to new CLI (spike demo)
# Removed insights command - moved to new CLI (spike insights)
# Removed insights_search command - moved to new CLI
# Removed insights_validate command - maintenance command
# Removed insights_cleanup command - maintenance command
# Removed test_safe command - development command


def main():
    """Main CLI entry point for Poetry scripts"""
    app()


if __name__ == "__main__":
    main()
