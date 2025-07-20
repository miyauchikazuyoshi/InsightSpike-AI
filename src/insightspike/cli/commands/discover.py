"""
Insight Discovery Command for InsightSpike CLI
============================================

Discovers hidden insights and connections in your knowledge base using
the geDIG (Graph Edit Distance + Information Gain) algorithm.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...core.error_handler import get_logger
from ...implementations.agents.main_agent import MainAgent

logger = get_logger("cli.discover")
console = Console()


class InsightDiscovery:
    """Handles insight discovery from knowledge base"""
    
    def __init__(self, agent: MainAgent):
        self.agent = agent
        self.insights_found = []
        
    def analyze_corpus(self, corpus_path: Path, file_extensions: List[str] = None) -> List[Path]:
        """Collect files from corpus directory"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.json']
            
        files = []
        if corpus_path.is_file():
            files = [corpus_path]
        else:
            for ext in file_extensions:
                files.extend(corpus_path.glob(f"**/*{ext}"))
                
        return files
    
    def load_corpus(self, files: List[Path], progress=None) -> int:
        """Load documents into agent's knowledge base"""
        loaded = 0
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                self.agent.add_knowledge(content)
                loaded += 1
                if progress:
                    progress.update(progress.task_ids[0], advance=1)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                if progress:
                    progress.update(progress.task_ids[0], advance=1)
        return loaded
    
    def discover_insights(self, min_spike: float = 0.7) -> List[Dict]:
        """Discover insights by analyzing the knowledge graph"""
        insights = []
        
        # Get current graph state from agent
        graph_state = self.agent.get_memory_graph_state()
        
        # Analyze recent reasoning cycles for spike patterns
        if hasattr(self.agent, 'reasoning_history'):
            for entry in self.agent.reasoning_history[-10:]:  # Last 10 queries
                if entry.get('quality', 0) > 0.8:
                    # High quality reasoning might indicate insight
                    insight = {
                        'type': 'reasoning_spike',
                        'spike_value': entry['quality'],
                        'question': entry['question'],
                        'confidence': entry['quality'],
                        'timestamp': time.time()
                    }
                    insights.append(insight)
        
        # Analyze graph metrics for structural insights
        if 'graph' in graph_state and graph_state['graph']:
            graph_metrics = graph_state['graph']
            # Check for significant graph changes (simplified spike detection)
            if graph_metrics.get('num_nodes', 0) > 0:
                # Look for patterns in the loaded knowledge
                insights.extend(self._analyze_knowledge_patterns(min_spike))
        else:
            # If no graph state, still try to analyze knowledge patterns
            insights.extend(self._analyze_knowledge_patterns(min_spike))
        
        # Sort by spike value
        insights.sort(key=lambda x: x.get('spike_value', 0), reverse=True)
        
        return insights
    
    def _analyze_knowledge_patterns(self, min_spike: float) -> List[Dict]:
        """Analyze loaded knowledge for patterns and insights"""
        insights = []
        
        # Get episodes from L2 memory
        if hasattr(self.agent, 'l2_memory') and hasattr(self.agent.l2_memory, 'episodes'):
            episodes = self.agent.l2_memory.episodes
            
            # Find semantic clusters
            if len(episodes) >= 3:
                # Simple pattern: look for related concepts
                concept_map = {}
                for ep in episodes:
                    # Extract key concepts (simplified)
                    words = ep.text.lower().split()
                    for word in words:
                        if len(word) > 4:  # Skip short words
                            concept_map[word] = concept_map.get(word, 0) + 1
                
                # Find concepts that appear in multiple episodes
                for concept, count in concept_map.items():
                    if count >= 2:  # Appears in at least 2 episodes
                        spike_value = min(1.0, count / len(episodes))
                        if spike_value >= min_spike:
                            insights.append({
                                'type': 'recurring_concept',
                                'spike_value': spike_value,
                                'description': f"Recurring concept '{concept}' found across {count} knowledge items",
                                'concept': concept,
                                'confidence': spike_value * 0.8,
                                'timestamp': time.time()
                            })
            
            # Find potential connections between concepts
            if len(episodes) >= 2:
                # Simple connection detection
                connections = []
                for i, ep1 in enumerate(episodes):
                    for j, ep2 in enumerate(episodes[i+1:], i+1):
                        # Check for shared concepts
                        words1 = set(w.lower() for w in ep1.text.split() if len(w) > 4)
                        words2 = set(w.lower() for w in ep2.text.split() if len(w) > 4)
                        shared = words1 & words2
                        
                        if len(shared) >= 2:  # At least 2 shared concepts
                            spike_value = min(1.0, len(shared) / 5)
                            if spike_value >= min_spike:
                                insights.append({
                                    'type': 'concept_bridge',
                                    'spike_value': spike_value,
                                    'description': f"Connection discovered between concepts: {', '.join(list(shared)[:3])}",
                                    'shared_concepts': list(shared),
                                    'confidence': spike_value * 0.7,
                                    'timestamp': time.time()
                                })
        
        return insights
    
    def find_concept_bridges(self, insights: List[Dict]) -> List[Dict]:
        """Identify bridge concepts from insights"""
        bridges = []
        
        # Simple heuristic: concepts that appear in multiple insights
        concept_frequency = {}
        for insight in insights:
            # Extract concepts from insight text
            text = insight.get('question', '') + ' ' + insight.get('description', '')
            # Simple word extraction (could be enhanced with NLP)
            words = text.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    concept_frequency[word] = concept_frequency.get(word, 0) + 1
        
        # Identify bridge concepts (appear in multiple contexts)
        for concept, freq in concept_frequency.items():
            if freq >= 2:
                bridges.append({
                    'concept': concept,
                    'frequency': freq,
                    'bridge_score': min(1.0, freq / 5.0)
                })
        
        bridges.sort(key=lambda x: x['bridge_score'], reverse=True)
        return bridges[:5]  # Top 5 bridge concepts


def discover_command(
    ctx: typer.Context,
    corpus: Optional[Path] = typer.Option(None, "--corpus", "-c", help="Path to document corpus"),
    min_spike: float = typer.Option(0.7, "--min-spike", "-s", help="Minimum spike threshold (0-1)"),
    max_insights: int = typer.Option(20, "--max-insights", "-m", help="Maximum insights to display"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Filter by categories (comma-separated)"),
    export_path: Optional[Path] = typer.Option(None, "--export", "-e", help="Export insights to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """
    Discover hidden insights and unexpected connections in your knowledge base.
    
    This command analyzes your knowledge graph to find:
    - Unexpected connections between concepts
    - Emergent patterns in your data
    - Bridge concepts that connect different domains
    - High-value insight spikes
    
    Examples:
        spike discover --corpus papers/ --min-spike 0.8
        spike discover --categories "causal,structural" --export insights.json
    """
    try:
        # Get agent from context
        from ..spike import DependencyFactory
        factory: DependencyFactory = ctx.obj
        
        # Show initialization message
        console.print("[yellow]Initializing InsightSpike (this may take a moment on first run)...[/yellow]")
        
        with console.status("[yellow]Loading AI components...[/yellow]"):
            agent = factory.get_agent("development")
        
        discovery = InsightDiscovery(agent)
        
        # If corpus provided, load it first
        if corpus:
            console.print(f"\n[blue]📚 Loading corpus from {corpus}...[/blue]")
            files = discovery.analyze_corpus(corpus)
            
            if not files:
                console.print("[yellow]No files found in corpus[/yellow]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Loading {len(files)} documents...", total=len(files))
                loaded = discovery.load_corpus(files, progress)
            
            console.print(f"[green]✅ Loaded {loaded} documents[/green]\n")
        
        # Discover insights
        console.print("[blue]🔍 Discovering insights...[/blue]\n")
        
        with console.status("Analyzing knowledge graph..."):
            insights = discovery.discover_insights(min_spike)
            
            # Filter by categories if specified
            if categories:
                cat_list = [c.strip() for c in categories.split(',')]
                insights = [i for i in insights if i.get('type') in cat_list]
            
            # Limit to max insights
            insights = insights[:max_insights]
            
            # Find bridge concepts
            bridges = discovery.find_concept_bridges(insights)
        
        # Display results
        if not insights:
            console.print("[yellow]No insights found above threshold[/yellow]")
            return
        
        console.print(f"[green bold]⚡ Discovered {len(insights)} insights[/green bold]\n")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            spike_value = insight.get('spike_value', 0)
            confidence = insight.get('confidence', 0)
            
            # Color based on spike value
            if spike_value >= 0.9:
                color = "red"
                emoji = "🔥"
            elif spike_value >= 0.8:
                color = "yellow"
                emoji = "⚡"
            else:
                color = "blue"
                emoji = "💡"
            
            panel_content = []
            if 'question' in insight:
                panel_content.append(f"Question: {insight['question']}")
            if 'description' in insight:
                panel_content.append(f"Description: {insight['description']}")
            
            panel_content.append(f"Confidence: {confidence:.0%}")
            
            panel = Panel(
                "\n".join(panel_content),
                title=f"{emoji} Insight #{i} [Spike: {spike_value:.2f}]",
                title_align="left",
                border_style=color
            )
            console.print(panel)
        
        # Display bridge concepts if found
        if bridges:
            console.print("\n[blue bold]🌉 Bridge Concepts:[/blue bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Concept", style="cyan")
            table.add_column("Frequency", justify="right")
            table.add_column("Bridge Score", justify="right")
            
            for bridge in bridges:
                table.add_row(
                    bridge['concept'],
                    str(bridge['frequency']),
                    f"{bridge['bridge_score']:.2f}"
                )
            
            console.print(table)
        
        # Summary statistics
        if verbose:
            console.print("\n[dim]📊 Summary Statistics:[/dim]")
            console.print(f"[dim]- Total insights: {len(insights)}[/dim]")
            console.print(f"[dim]- Average spike: {sum(i['spike_value'] for i in insights)/len(insights):.3f}[/dim]")
            console.print(f"[dim]- Insight types: {set(i.get('type', 'unknown') for i in insights)}[/dim]")
        
        # Export if requested
        if export_path:
            export_data = {
                'timestamp': time.time(),
                'parameters': {
                    'min_spike': min_spike,
                    'max_insights': max_insights,
                    'categories': categories
                },
                'insights': insights,
                'bridge_concepts': bridges,
                'summary': {
                    'total_insights': len(insights),
                    'average_spike': sum(i['spike_value'] for i in insights)/len(insights) if insights else 0
                }
            }
            
            export_path.write_text(json.dumps(export_data, indent=2))
            console.print(f"\n[green]✅ Exported to {export_path}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)