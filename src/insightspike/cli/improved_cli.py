#!/usr/bin/env python3
"""
Improved CLI for InsightSpike - User-friendly and intuitive
==========================================================

Features:
- Simple, memorable commands
- Better error messages
- Interactive mode
- Configuration management
- Progress indicators
"""

import sys
import json
from pathlib import Path
from typing import Optional, List
import time

import typer
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

# Import our simplified config system
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from insightspike.config import ConfigManager, ConfigPresets, SimpleConfig, get_config
from insightspike.core.agents.main_agent import MainAgent
from insightspike.utils.error_handler import get_logger, InsightSpikeError

app = typer.Typer(
    name="spike",
    help="InsightSpike AI - Discover insights through knowledge synthesis",
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()
logger = get_logger("cli")


# Global state for interactive mode
class CLIState:
    def __init__(self):
        self.agent: Optional[MainAgent] = None
        self.config_manager: Optional[ConfigManager] = None
        self.interactive_mode: bool = False


state = CLIState()


def get_or_create_agent(preset: str = "development") -> MainAgent:
    """Get existing agent or create new one"""
    if state.agent is None:
        with console.status("[yellow]Initializing AI agent...[/yellow]"):
            # Get configuration
            if state.config_manager is None:
                config = get_config(preset)
                state.config_manager = ConfigManager(config)

            # Convert to legacy format
            legacy_config = state.config_manager.to_legacy_config()

            # Create and initialize agent
            state.agent = MainAgent(config=legacy_config)
            if not state.agent.initialize():
                console.print("[red]Failed to initialize agent[/red]")
                raise typer.Exit(code=1)

    return state.agent


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

        # Get or create agent
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

        # Display response
        console.print(f"\n[bold green]💡 Answer:[/bold green]")
        console.print(result.get("response", "No response generated"))

        # Show metrics if verbose
        if verbose:
            console.print(f"\n[dim]Quality: {result.get('reasoning_quality', 0):.3f}")
            console.print(f"Cycles: {result.get('total_cycles', 0)}")
            console.print(
                f"Spike detected: {result.get('spike_detected', False)}[/dim]"
            )

        # Show insight indicator
        if result.get("spike_detected", False):
            console.print("\n[bold yellow]⚡ Insight spike detected![/bold yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def embed(path: Path, preset: str = typer.Option("development", help="Config preset")):
    """Embed documents into the knowledge base (file or directory)"""
    try:
        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(code=1)

        # Count documents
        if path.is_file():
            files = [path] if path.suffix == ".txt" else []
        else:
            files = list(path.glob("**/*.txt"))

        if not files:
            console.print("[yellow]No text files found[/yellow]")
            return

        console.print(f"[blue]📚 Found {len(files)} document(s)[/blue]")

        # Get agent
        agent = get_or_create_agent(preset)

        # Process files with progress
        added = 0
        with Progress(console=console) as progress:
            task = progress.add_task("Embedding...", total=len(files))

            for file in files:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()

                    if agent.add_episode_with_graph_update(text=content):
                        added += 1

                except Exception as e:
                    logger.warning(f"Failed to process {file}: {e}")

                progress.update(task, advance=1)

        console.print(f"[green]✅ Embedded {added}/{len(files)} documents[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, set, save, load, preset"),
    key: Optional[str] = typer.Argument(None, help="Config key (for set)"),
    value: Optional[str] = typer.Argument(None, help="Config value (for set)"),
):
    """Manage configuration settings"""
    try:
        if state.config_manager is None:
            state.config_manager = ConfigManager()

        if action == "show":
            # Show current configuration
            table = Table(title="Current Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            config_dict = state.config_manager.config.to_dict()
            for k, v in config_dict.items():
                if not k.startswith("_"):
                    table.add_row(k, str(v))

            console.print(table)

        elif action == "set" and key and value:
            # Set configuration value
            try:
                # Type conversion
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)

                state.config_manager.set(key, value)
                console.print(f"[green]✅ Set {key} = {value}[/green]")

                # Reinitialize agent if needed
                if state.agent is not None:
                    state.agent = None
                    console.print(
                        "[yellow]Agent will be reinitialized with new config[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]Error setting config: {e}[/red]")

        elif action == "save" and key:
            # Save configuration to file
            state.config_manager.config.save(key)
            console.print(f"[green]✅ Configuration saved to {key}[/green]")

        elif action == "load" and key:
            # Load configuration from file
            loaded_config = SimpleConfig.load(key)
            state.config_manager = ConfigManager(loaded_config)
            state.agent = None  # Reset agent
            console.print(f"[green]✅ Configuration loaded from {key}[/green]")

        elif action == "preset" and key:
            # Load preset configuration
            presets = ["development", "testing", "production", "experiment", "cloud"]
            if key not in presets:
                console.print(f"[red]Unknown preset: {key}[/red]")
                console.print(f"Available presets: {', '.join(presets)}")
                raise typer.Exit(code=1)

            preset_func = getattr(ConfigPresets, key)
            state.config_manager = ConfigManager(preset_func())
            state.agent = None  # Reset agent
            console.print(f"[green]✅ Loaded {key} preset[/green]")

        else:
            console.print("[red]Invalid config command[/red]")
            console.print("Usage:")
            console.print("  spike config show")
            console.print("  spike config set <key> <value>")
            console.print("  spike config save <filename>")
            console.print("  spike config load <filename>")
            console.print("  spike config preset <preset_name>")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def chat():
    """Start interactive chat mode"""
    console.print("[bold blue]🤖 InsightSpike Interactive Mode[/bold blue]")
    console.print("Type 'help' for commands, 'exit' to quit\n")

    state.interactive_mode = True

    # Initialize agent
    agent = get_or_create_agent()

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break

            elif user_input.lower() == "help":
                console.print("\n[bold]Available commands:[/bold]")
                console.print("  help     - Show this help")
                console.print("  stats    - Show agent statistics")
                console.print("  clear    - Clear conversation history")
                console.print("  config   - Show current configuration")
                console.print("  exit     - Exit chat mode\n")
                continue

            elif user_input.lower() == "stats":
                show_stats()
                continue

            elif user_input.lower() == "clear":
                if Confirm.ask("Clear conversation history?"):
                    state.agent = None
                    agent = get_or_create_agent()
                    console.print("[green]History cleared[/green]\n")
                continue

            elif user_input.lower() == "config":
                config("show")
                continue

            # Process as question
            with console.status("[yellow]Thinking...[/yellow]"):
                result = agent.process_question(user_input, max_cycles=3, verbose=False)

            # Display response
            console.print(f"\n[bold green]InsightSpike:[/bold green]")
            console.print(result.get("response", "No response generated"))

            # Show insight indicator
            if result.get("spike_detected", False):
                console.print("[bold yellow]⚡ Insight spike detected![/bold yellow]")

            console.print()  # Empty line for readability

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Chat error: {e}", exc_info=True)


@app.command()
def stats():
    """Show agent statistics and insights"""
    show_stats()


def show_stats():
    """Display agent statistics"""
    try:
        agent = get_or_create_agent()
        stats = agent.get_stats()

        # Create stats table
        table = Table(title="Agent Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Initialized", "✅" if stats.get("initialized", False) else "❌")
        table.add_row("Total cycles", str(stats.get("total_cycles", 0)))
        table.add_row(
            "Reasoning history", str(stats.get("reasoning_history_length", 0))
        )
        table.add_row("Average quality", f"{stats.get('average_quality', 0):.3f}")

        # Memory stats
        memory_stats = stats.get("memory_stats", {})
        table.add_row("Total episodes", str(memory_stats.get("total_episodes", 0)))
        table.add_row("Total documents", str(memory_stats.get("total_documents", 0)))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")


@app.command()
def experiment(
    name: str = typer.Option("simple", help="Experiment: simple, insight, math"),
    episodes: int = typer.Option(10, help="Number of episodes to run"),
):
    """Run experiments to demonstrate capabilities"""
    try:
        console.print(f"[bold blue]🧪 Running {name} experiment[/bold blue]")

        if name == "simple":
            run_simple_experiment(episodes)
        elif name == "insight":
            run_insight_experiment(episodes)
        elif name == "math":
            run_math_experiment(episodes)
        else:
            console.print(f"[red]Unknown experiment: {name}[/red]")
            console.print("Available: simple, insight, math")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def demo():
    """Run interactive demo showcasing InsightSpike capabilities"""
    console.print("[bold blue]🧠 InsightSpike Demo[/bold blue]")
    console.print(
        "This demo shows how InsightSpike detects insights through knowledge synthesis\n"
    )

    # Initialize with experiment preset for real responses
    agent = get_or_create_agent("experiment")

    # Demo knowledge base
    demo_knowledge = [
        "量子コンピュータは重ね合わせの原理を利用する。",
        "古典コンピュータはビットで0か1の状態を持つ。",
        "エントロピーは系の無秩序さを表す。",
        "情報理論では情報量をビットで測定する。",
        "機械学習は大量のデータからパターンを学習する。",
        "脳のニューロンは電気信号で情報を伝達する。",
    ]

    # Add knowledge
    console.print("[yellow]📚 Adding knowledge base...[/yellow]")
    with Progress(console=console) as progress:
        task = progress.add_task("Loading knowledge...", total=len(demo_knowledge))

        for knowledge in demo_knowledge:
            agent.add_episode_with_graph_update(text=knowledge)
            progress.update(task, advance=1)

    console.print("[green]✅ Knowledge base loaded[/green]\n")

    # Demo questions
    demo_questions = [
        ("量子コンピュータと情報理論の関係は？", "Cross-domain synthesis"),
        ("機械学習と脳の学習の共通点は？", "Analogical reasoning"),
        ("エントロピーと情報量の関係を説明してください。", "Conceptual integration"),
    ]

    console.print("[bold]Demonstrating insight detection:[/bold]\n")

    for question, insight_type in demo_questions:
        console.print(f"[cyan]Q:[/cyan] {question}")
        console.print(f"[dim]Expected: {insight_type}[/dim]")

        with console.status("[yellow]Thinking...[/yellow]"):
            result = agent.process_question(question, max_cycles=3)

        response = result.get("response", "No response")
        console.print(
            f"[green]A:[/green] {response[:200]}{'...' if len(response) > 200 else ''}"
        )

        if result.get("spike_detected", False):
            console.print("[bold yellow]⚡ Insight spike detected![/bold yellow]")

        console.print()

    console.print("[bold green]✅ Demo complete![/bold green]")
    console.print(
        '\nTry querying your own questions with: [bold]spike query "Your question"[/bold]'
    )


@app.command()
def insights(limit: int = typer.Option(5, help="Number of insights to show")):
    """Show discovered insights and statistics"""
    try:
        from insightspike.detection.insight_registry import InsightFactRegistry

        registry = InsightFactRegistry()
        stats = registry.get_optimization_stats()
        total_insights = len(registry.insights)

        # Create stats table
        table = Table(title="Insight Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Insights", str(total_insights))
        table.add_row("Average Quality", f"{stats.get('avg_quality', 0):.3f}")
        table.add_row("Average GED", f"{stats.get('avg_ged', 0):.3f}")
        table.add_row("Average IG", f"{stats.get('avg_ig', 0):.3f}")

        console.print(table)

        if total_insights > 0:
            # Show recent insights
            console.print(
                f"\n[bold]Recent Insights (showing {min(limit, total_insights)}):[/bold]\n"
            )

            recent_insights = sorted(
                registry.insights.values(), key=lambda x: x.generated_at, reverse=True
            )[:limit]

            for i, insight in enumerate(recent_insights, 1):
                console.print(f"[bold]{i}. {insight.relationship_type.title()}[/bold]")
                console.print(
                    f"   {insight.text[:150]}{'...' if len(insight.text) > 150 else ''}"
                )
                console.print(
                    f"   [dim]Quality: {insight.quality_score:.3f}, GED: {insight.ged_optimization:.3f}[/dim]\n"
                )
        else:
            console.print(
                "\n[yellow]No insights discovered yet. Try asking some questions![/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error accessing insights: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("insights-search")
def insights_search(
    concept: str, limit: int = typer.Option(10, help="Maximum results to show")
):
    """Search for insights related to a concept"""
    try:
        from insightspike.detection.insight_registry import InsightFactRegistry

        registry = InsightFactRegistry()
        relevant_insights = registry.find_relevant_insights(
            [concept.lower()], limit=limit
        )

        console.print(f"[bold blue]🔍 Insights about '{concept}'[/bold blue]\n")

        if relevant_insights:
            for i, insight in enumerate(relevant_insights, 1):
                console.print(f"[bold]{i}. {insight.relationship_type.title()}[/bold]")
                console.print(f"   {insight.text}")
                console.print(f"   [dim]Quality: {insight.quality_score:.3f}[/dim]\n")
        else:
            console.print(f"[yellow]No insights found about '{concept}'[/yellow]")
            console.print(
                "Try asking questions about this concept to discover insights."
            )

    except Exception as e:
        console.print(f"[red]Error searching insights: {e}[/red]")
        raise typer.Exit(code=1)


def run_simple_experiment(episodes: int):
    """Run simple spike detection experiment"""
    agent = get_or_create_agent("experiment")

    # Sample episodes that should trigger spikes
    sample_episodes = [
        "システムAは独立して動作する。",
        "システムBも独立して動作する。",
        "AとBを統合すると新しい性質が生まれる。",  # Spike expected
        "この統合により、全体の効率が向上する。",
        "量子コンピュータは重ね合わせを利用する。",
        "古典コンピュータはビットを使用する。",
        "量子と古典の融合が次世代計算を実現する。",  # Spike expected
    ]

    spikes_detected = 0

    with Progress(console=console) as progress:
        task = progress.add_task(
            "Running experiment...", total=min(episodes, len(sample_episodes))
        )

        for i in range(min(episodes, len(sample_episodes))):
            episode = sample_episodes[i]
            result = agent.add_episode_with_graph_update(text=episode)

            if result.get("graph_analysis", {}).get("spike_detected", False):
                spikes_detected += 1
                console.print(f"\n[yellow]⚡ Spike detected:[/yellow] {episode[:50]}...")

            progress.update(task, advance=1)

    console.print(f"\n[green]Experiment complete![/green]")
    console.print(f"Episodes: {min(episodes, len(sample_episodes))}")
    console.print(f"Spikes detected: {spikes_detected}")


def run_insight_experiment(episodes: int):
    """Run insight synthesis experiment"""
    agent = get_or_create_agent("experiment")

    # Knowledge base
    knowledge = [
        "エントロピーは系の無秩序さを表す尺度である。",
        "情報理論では、情報量はエントロピーで測定される。",
        "生命システムは局所的にエントロピーを減少させる。",
        "機械学習モデルは情報を圧縮して表現を学習する。",
        "創発とは、個々の要素から予測できない性質が生まれることである。",
    ]

    # Add knowledge
    console.print("[yellow]Adding knowledge base...[/yellow]")
    for k in knowledge[:episodes]:
        agent.add_episode_with_graph_update(text=k)

    # Test questions requiring synthesis
    questions = [
        "生命と情報理論の関係は何ですか？",
        "機械学習と創発現象の共通点は？",
        "エントロピーと学習の関係を説明してください。",
    ]

    console.print("\n[blue]Testing synthesis capabilities:[/blue]")
    for q in questions[: min(3, episodes // 2)]:
        console.print(f"\n[cyan]Q:[/cyan] {q}")
        result = agent.process_question(q, max_cycles=3)
        console.print(
            f"[green]A:[/green] {result.get('response', 'No response')[:150]}..."
        )

        if result.get("spike_detected", False):
            console.print("[yellow]⚡ Insight spike detected![/yellow]")


def run_math_experiment(episodes: int):
    """Run mathematical foundations experiment"""
    agent = get_or_create_agent("experiment")

    # Mathematical concepts
    math_concepts = [
        "群論は対称性と変換を数学的に扱う。",
        "位相空間は連続性の概念を一般化する。",
        "カテゴリー理論は数学的構造間の関係を研究する。",
        "情報幾何学は確率分布の空間に幾何学的構造を与える。",
        "力学系理論はシステムの時間発展を研究する。",
    ]

    # Add concepts
    console.print("[yellow]Adding mathematical concepts...[/yellow]")
    for concept in math_concepts[:episodes]:
        agent.add_episode_with_graph_update(text=concept)

    # Test cross-domain synthesis
    console.print("\n[blue]Testing mathematical synthesis:[/blue]")
    result = agent.process_question("群論と情報理論はどのように関連していますか？", max_cycles=3)

    console.print(
        f"\n[green]Synthesis:[/green] {result.get('response', 'No response')}"
    )
    if result.get("spike_detected", False):
        console.print("[yellow]⚡ Mathematical insight detected![/yellow]")


@app.command()
def version():
    """Show version information"""
    console.print("[bold]InsightSpike AI[/bold]")
    console.print("Version: 0.2.0 (Improved CLI)")
    console.print("Python: " + sys.version.split()[0])

    # Show config info
    if state.config_manager:
        config = state.config_manager.config
        console.print(f"\nActive config:")
        console.print(f"  Mode: {config.mode}")
        console.print(f"  Safe mode: {config.safe_mode}")
        console.print(f"  Model: {config.llm_model}")


# Aliases for common commands
app.command("q")(query)  # spike q "question"
app.command("c")(chat)  # spike c (start chat)
app.command("e")(embed)  # spike e path/to/docs

# Legacy aliases for backward compatibility
app.command("ask")(query)  # spike ask → query
app.command("learn")(embed)  # spike learn → embed
app.command("l")(embed)  # spike l → embed


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
