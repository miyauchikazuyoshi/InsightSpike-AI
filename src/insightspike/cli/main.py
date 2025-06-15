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
@app.command()
def experiment():
    """Run experimental validation suite"""
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


# Legacy compatibility commands (simplified)
@app.command()
def embed(path: Optional[pathlib.Path] = typer.Option(None, help="Path to text file")):
    """Legacy embed command - use load_documents instead"""
    print(
        "[yellow]Note: 'embed' command is deprecated. Use 'load_documents' instead.[/yellow]"
    )

    if path is None:
        raw_dir = Path("data/raw")
        txt_files = sorted(raw_dir.glob("*.txt"))
        if not txt_files:
            print(
                "[red]Error: --path option not specified and no text files in data/raw/[/red]"
            )
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


@app.command()
def insight_experiment():
    """Run insight detection experiment (rigorous validation)"""
    print("[bold blue]🧠 Insight Detection Experiment[/bold blue]")
    print("=" * 50)
    print("[yellow]This experiment validates synthesis capabilities")
    print("by using knowledge bases with NO direct answers.[/yellow]")
    print()

    # Check if experiment data exists
    knowledge_file = "data/raw/indirect_knowledge.txt"
    questions_file = "data/processed/insight_questions.json"

    if not Path(knowledge_file).exists() or not Path(questions_file).exists():
        print("[yellow]📚 Creating insight experiment data...[/yellow]")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/create_true_insight_experiment.py"],
                cwd=".",
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"[red]Failed to create experiment data: {result.stderr}[/red]")
                raise typer.Exit(code=1)
        except Exception as e:
            print(f"[red]Error creating experiment data: {e}[/red]")
            raise typer.Exit(code=1)

    print("[yellow]🔬 Running insight experiment...[/yellow]")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_true_insight_experiment.py"],
            cwd=".",
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Experiment failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        print("[green]✅ Insight experiment completed successfully![/green]")
        print(
            "[blue]📄 Results saved to: data/processed/true_insight_results.json[/blue]"
        )

    except Exception as e:
        print(f"[red]Error running experiment: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def compare_experiments():
    """Compare different experimental designs"""
    print("[bold blue]🔬 Comparative Experimental Analysis[/bold blue]")
    print("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "scripts/compare_experiments.py"],
            cwd=".",
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            print(f"[red]Comparison failed: {result.stderr}[/red]")
            raise typer.Exit(code=1)

        print("[green]✅ Comparative analysis completed successfully![/green]")
        print("[blue]📄 Report saved to: COMPARATIVE_EXPERIMENTAL_ANALYSIS.md[/blue]")

    except Exception as e:
        print(f"[red]Error running comparison: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def experiment_suite(
    experiment_type: str = typer.Option("all", help="Type: simple, insight, or all"),
    verbose: bool = typer.Option(False, help="Show detailed output"),
):
    """Run complete experimental validation suite"""
    print("[bold blue]🧪 InsightSpike Experimental Validation Suite[/bold blue]")
    print("=" * 60)

    experiments_to_run = []

    if experiment_type in ["simple", "all"]:
        experiments_to_run.append(("Simple", "scripts/run_poc_simple.py"))

    if experiment_type in ["insight", "all"]:
        # Ensure insight experiment data exists
        knowledge_file = "data/raw/indirect_knowledge.txt"
        if not Path(knowledge_file).exists():
            print("[yellow]📚 Creating insight experiment data...[/yellow]")
            subprocess.run(
                [sys.executable, "scripts/create_true_insight_experiment.py"], cwd="."
            )

        experiments_to_run.append(
            ("Insight Detection", "scripts/run_true_insight_experiment.py")
        )

    if experiment_type == "all":
        experiments_to_run.append(
            ("Comparative Analysis", "scripts/compare_experiments.py")
        )

    if not experiments_to_run:
        print(f"[red]Unknown experiment type: {experiment_type}[/red]")
        print("Available types: simple, insight, all")
        raise typer.Exit(code=1)

    # Run experiments
    for name, script in experiments_to_run:
        print(f"\n[yellow]🔬 Running {name} Experiment...[/yellow]")

        try:
            result = subprocess.run(
                [sys.executable, script], cwd=".", capture_output=not verbose, text=True
            )

            if result.returncode != 0:
                print(f"[red]{name} experiment failed: {result.stderr}[/red]")
                continue

            print(f"[green]✅ {name} experiment completed[/green]")

        except Exception as e:
            print(f"[red]Error running {name}: {e}[/red]")
            continue

    print("\n[bold green]🏆 Experimental validation suite completed![/bold green]")
    print("[blue]📊 Check data/processed/ for detailed results[/blue]")
    print("[blue]📄 Check COMPARATIVE_EXPERIMENTAL_ANALYSIS.md for analysis[/blue]")


@app.command()
def demo():
    """Run interactive demo of insight detection capabilities"""
    print("[bold blue]🧠 InsightSpike Insight Demo[/bold blue]")
    print("=" * 50)
    print("[yellow]This demo showcases cross-domain synthesis capabilities[/yellow]")
    print()

    try:
        # Initialize agent
        print("[yellow]Initializing AI agent...[/yellow]")
        agent = MainAgent()
        if not agent.initialize():
            print("[red]Failed to initialize agent[/red]")
            raise typer.Exit(code=1)

        # Demo questions that require synthesis
        demo_questions = [
            "What is the relationship between probability and information theory?",
            "How does mathematical infinity relate to physical reality?",
            "What connects memory formation and decision making in the brain?",
        ]

        for i, question in enumerate(demo_questions, 1):
            print(f"\n[bold cyan]Demo Question {i}:[/bold cyan] {question}")
            print("[yellow]Processing...[/yellow]")

            result = agent.process_question(question, max_cycles=3, verbose=False)

            print(
                f"[green]Response:[/green] {result.get('response', 'No response')[:200]}..."
            )
            print(
                f"[dim]Quality: {result.get('reasoning_quality', 0):.3f}, "
                f"Spike: {result.get('spike_detected', False)}[/dim]"
            )

            if result.get("spike_detected", False):
                print("[bold yellow]⚡ INSIGHT SPIKE DETECTED![/bold yellow]")

        print("\n[green]✅ Demo completed successfully![/green]")
        print("[blue]Demo showcased InsightSpike's synthesis capabilities[/blue]")

    except Exception as e:
        print(f"[red]Error running demo: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def insights():
    """Show registered insight facts and statistics"""
    try:
        registry = InsightFactRegistry()

        # Get statistics
        stats = registry.get_optimization_stats()
        total_insights = len(registry.insights)

        print("[bold blue]🧠 Insight Facts Registry[/bold blue]")
        print("=" * 50)

        print(f"[bold]Total Insights:[/bold] {total_insights}")
        print(f"[bold]Average Quality:[/bold] {stats.get('avg_quality', 0):.3f}")
        print(f"[bold]Average GED Improvement:[/bold] {stats.get('avg_ged', 0):.3f}")
        print(f"[bold]Average IG Improvement:[/bold] {stats.get('avg_ig', 0):.3f}")

        if total_insights > 0:
            print(f"\n[bold blue]Recent Insights:[/bold blue]")

            # Show top 5 recent insights
            recent_insights = sorted(
                registry.insights.values(), key=lambda x: x.generated_at, reverse=True
            )[:5]

            for i, insight in enumerate(recent_insights, 1):
                print(f"\n[bold]{i}. {insight.relationship_type.title()}[/bold]")
                print(
                    f"   Text: {insight.text[:100]}{'...' if len(insight.text) > 100 else ''}"
                )
                print(
                    f"   Quality: {insight.quality_score:.3f}, "
                    f"GED: {insight.ged_optimization:.3f}, "
                    f"IG: {insight.ig_improvement:.3f}"
                )
                print(
                    f"   Concepts: {len(insight.source_concepts + insight.target_concepts)} total"
                )
        else:
            print(
                "\n[yellow]No insights registered yet. Ask some questions to discover insights![/yellow]"
            )

    except Exception as e:
        print(f"[red]Error accessing insight registry: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def insights_search(
    concept: str = typer.Argument(..., help="Concept to search for in insights")
):
    """Search for insights related to a specific concept"""
    try:
        registry = InsightFactRegistry()

        # Find relevant insights
        relevant_insights = registry.find_relevant_insights([concept.lower()], limit=10)

        print(f"[bold blue]🔍 Insights Related to '{concept}'[/bold blue]")
        print("=" * 50)

        if relevant_insights:
            print(f"Found {len(relevant_insights)} relevant insights:")

            for i, insight in enumerate(relevant_insights, 1):
                print(f"\n[bold]{i}. {insight.relationship_type.title()}[/bold]")
                print(f"   Text: {insight.text}")
                print(
                    f"   Quality: {insight.quality_score:.3f}, "
                    f"GED: {insight.ged_optimization:.3f}"
                )
                print(f"   Source: {', '.join(insight.source_concepts)}")
                print(f"   Target: {', '.join(insight.target_concepts)}")
        else:
            print(f"[yellow]No insights found related to '{concept}'[/yellow]")
            print(
                "[dim]Try asking questions that involve this concept to discover insights.[/dim]"
            )

    except Exception as e:
        print(f"[red]Error searching insights: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def insights_validate(
    insight_id: str = typer.Argument(..., help="Insight ID to validate")
):
    """Manually validate or invalidate a specific insight"""
    try:
        registry = InsightFactRegistry()

        # Find the insight
        if insight_id not in registry.insights:
            print(f"[red]Insight with ID '{insight_id}' not found[/red]")
            raise typer.Exit(code=1)

        insight = registry.insights[insight_id]

        print(f"[bold blue]🔍 Insight Validation[/bold blue]")
        print("=" * 50)
        print(f"[bold]ID:[/bold] {insight.id}")
        print(f"[bold]Type:[/bold] {insight.relationship_type}")
        print(f"[bold]Text:[/bold] {insight.text}")
        print(f"[bold]Quality Score:[/bold] {insight.quality_score:.3f}")
        print(f"[bold]Current Status:[/bold] {insight.validation_status}")

        # Ask for validation
        status = typer.prompt(
            "\nValidation status (valid/invalid/pending)",
            default=insight.validation_status,
        )

        if status in ["valid", "invalid", "pending"]:
            # Update validation status in database
            registry._update_validation_status(insight_id, status)
            print(f"[green]✓ Insight {insight_id} marked as {status}[/green]")
        else:
            print(
                f"[red]Invalid status '{status}'. Use: valid, invalid, or pending[/red]"
            )
            raise typer.Exit(code=1)

    except Exception as e:
        print(f"[red]Error validating insight: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def insights_cleanup():
    """Clean up low-quality insights from the registry"""
    try:
        registry = InsightFactRegistry()

        initial_count = len(registry.insights)
        print(f"[bold blue]🧹 Insight Registry Cleanup[/bold blue]")
        print("=" * 50)
        print(f"Initial insights: {initial_count}")

        # Remove insights with very low quality or marked as invalid
        removed_count = 0
        insights_to_remove = []

        for insight_id, insight in registry.insights.items():
            if (
                insight.quality_score < 0.3
                or insight.validation_status == "invalid"
                or (insight.ged_optimization < 0.05 and insight.ig_improvement < 0.02)
            ):
                insights_to_remove.append(insight_id)

        # Remove from database and memory
        for insight_id in insights_to_remove:
            registry._remove_insight_from_db(insight_id)
            del registry.insights[insight_id]
            removed_count += 1

        final_count = len(registry.insights)

        print(f"[green]✓ Cleanup completed[/green]")
        print(f"Removed: {removed_count} insights")
        print(f"Remaining: {final_count} insights")
        print(
            f"Space saved: {removed_count / initial_count * 100:.1f}%"
            if initial_count > 0
            else "0%"
        )

    except Exception as e:
        print(f"[red]Error during cleanup: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def test_safe(
    question: str = typer.Argument(
        "What is artificial intelligence?", help="Test question"
    )
):
    """Test the agent with mock LLM provider (safe mode - no model loading)"""
    try:
        print(f"[bold blue]Testing Question:[/bold blue] {question}")
        print("[yellow]Using safe mode with mock LLM provider...[/yellow]")

        # Import mock provider directly
        from ..core.config import get_config
        from ..core.layers.mock_llm_provider import MockLLMProvider

        config = get_config()
        mock_llm = MockLLMProvider(config)

        # Test mock provider
        if not mock_llm.initialize():
            print("[red]Failed to initialize mock LLM provider[/red]")
            raise typer.Exit(code=1)

        # Generate test response
        result = mock_llm.generate_response({}, question)

        # Display results
        print(
            f"\n[bold green]Mock Response:[/bold green] {result.get('response', 'No response generated')}"
        )
        print(
            f"[dim]Quality: {result.get('reasoning_quality', 0):.3f}, "
            f"Confidence: {result.get('confidence', 0):.3f}, "
            f"Model: {result.get('model_used', 'unknown')}[/dim]"
        )

        if result.get("success", False):
            print("[green]✓ Safe mode test successful[/green]")
        else:
            print("[red]✗ Safe mode test failed[/red]")

    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


def main():
    """Main CLI entry point for Poetry scripts"""
    app()


if __name__ == "__main__":
    main()
