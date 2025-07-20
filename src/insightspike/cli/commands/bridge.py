"""
spike bridge command - Find conceptual bridges between ideas
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
import numpy as np

from ...implementations.agents.main_agent import MainAgent as SimpleRAGGraph

logger = logging.getLogger(__name__)
console = Console()


class ConceptBridge:
    """Find conceptual paths between seemingly unrelated ideas"""
    
    def __init__(self, agent: SimpleRAGGraph):
        self.agent = agent
        self.l2_memory = agent.l2_memory
        self.l3_reasoner = agent.l3_graph
        
    def find_bridges(
        self, 
        concept1: str, 
        concept2: str, 
        max_hops: int = 3,
        top_k: int = 5
    ) -> List[Dict]:
        """Find bridge concepts connecting two ideas"""
        
        # Embed concepts
        vec1 = self.l2_memory._get_embedding(concept1)
        vec2 = self.l2_memory._get_embedding(concept2)
        
        # Find nearest neighbors for each concept
        neighbors1 = self._find_neighbors(vec1, k=10)
        neighbors2 = self._find_neighbors(vec2, k=10)
        
        # Find paths between concepts
        paths = self._find_paths(
            concept1, concept2,
            neighbors1, neighbors2,
            max_hops=max_hops
        )
        
        # Rank paths by quality
        ranked_paths = self._rank_paths(paths, vec1, vec2)
        
        return ranked_paths[:top_k]
    
    def _find_neighbors(self, vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Find nearest neighbors in the knowledge base"""
        if not hasattr(self.l2_memory, 'faiss_index') or self.l2_memory.faiss_index is None:
            return []
            
        # Search FAISS index
        D, I = self.l2_memory.faiss_index.search(
            vector.reshape(1, -1).astype(np.float32), 
            k
        )
        
        neighbors = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.l2_memory.episodes):
                episode = self.l2_memory.episodes[idx]
                neighbors.append({
                    'content': episode.text,
                    'distance': float(dist),
                    'index': int(idx)
                })
        
        return neighbors
    
    def _find_paths(
        self,
        start: str,
        end: str,
        neighbors1: List[Dict],
        neighbors2: List[Dict],
        max_hops: int
    ) -> List[List[str]]:
        """Find paths between concepts using BFS"""
        paths = []
        
        # Direct connection check
        for n1 in neighbors1:
            for n2 in neighbors2:
                if n1['index'] == n2['index']:
                    # Found direct bridge
                    paths.append([start, n1['content'], end])
        
        # Multi-hop paths
        if max_hops > 1:
            # Find intermediate concepts
            visited = set()
            queue = [(start, [start], 0)]
            
            while queue:
                current, path, hops = queue.pop(0)
                
                if hops >= max_hops:
                    continue
                
                # Get neighbors of current concept
                current_vec = self.l2_memory._get_embedding(current)
                current_neighbors = self._find_neighbors(current_vec, k=5)
                
                for neighbor in current_neighbors:
                    content = neighbor['content']
                    
                    if content in visited or content in path:
                        continue
                    
                    new_path = path + [content]
                    
                    # Check if we can reach the end
                    if self._concepts_similar(content, end):
                        paths.append(new_path + [end])
                    else:
                        queue.append((content, new_path, hops + 1))
                        visited.add(content)
        
        return paths
    
    def _concepts_similar(self, concept1: str, concept2: str, threshold: float = 0.7) -> bool:
        """Check if two concepts are similar enough"""
        vec1 = self.l2_memory._get_embedding(concept1)
        vec2 = self.l2_memory._get_embedding(concept2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity > threshold
    
    def _rank_paths(self, paths: List[List[str]], vec1: np.ndarray, vec2: np.ndarray) -> List[Dict]:
        """Rank paths by quality metrics"""
        ranked = []
        
        for path in paths:
            # Calculate path metrics
            length = len(path)
            coherence = self._calculate_path_coherence(path)
            novelty = self._calculate_novelty(path, vec1, vec2)
            
            # Combined score
            score = (1.0 / length) * coherence * novelty
            
            ranked.append({
                'path': path,
                'length': length,
                'coherence': coherence,
                'novelty': novelty,
                'score': score,
                'explanation': self._generate_explanation(path)
            })
        
        # Sort by score
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """Calculate how coherent a path is"""
        if len(path) < 2:
            return 1.0
        
        coherences = []
        for i in range(len(path) - 1):
            vec1 = self.l2_memory._get_embedding(path[i])
            vec2 = self.l2_memory._get_embedding(path[i+1])
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            coherences.append(similarity)
        
        return np.mean(coherences)
    
    def _calculate_novelty(self, path: List[str], vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate how novel/unexpected the bridge concepts are"""
        if len(path) <= 2:
            return 0.1  # Direct connection is not novel
        
        # Get bridge concepts (exclude start and end)
        bridges = path[1:-1]
        
        novelties = []
        for bridge in bridges:
            bridge_vec = self.l2_memory._get_embedding(bridge)
            
            # Distance from both endpoints
            dist1 = np.linalg.norm(bridge_vec - vec1)
            dist2 = np.linalg.norm(bridge_vec - vec2)
            
            # Higher distance = more novel
            novelty = (dist1 + dist2) / 2
            novelties.append(novelty)
        
        return np.mean(novelties) if novelties else 0.1
    
    def _generate_explanation(self, path: List[str]) -> str:
        """Generate human-readable explanation of the path"""
        if len(path) == 2:
            return f"Direct connection: {path[0]} ↔ {path[1]}"
        elif len(path) == 3:
            return f"Bridge through '{path[1]}': {path[0]} → {path[1]} → {path[2]}"
        else:
            bridges = " → ".join(path[1:-1])
            return f"Multi-hop path: {path[0]} → [{bridges}] → {path[-1]}"


def visualize_bridge_path(bridge_result: Dict) -> None:
    """Visualize a bridge path with Rich"""
    path = bridge_result['path']
    
    # Create tree visualization
    tree = Tree(f"[bold cyan]{path[0]}[/bold cyan] (Start)")
    
    current = tree
    for i, concept in enumerate(path[1:-1], 1):
        current = current.add(f"[yellow]→ {concept}[/yellow] (Bridge {i})")
    
    current.add(f"[bold green]{path[-1]}[/bold green] (End)")
    
    # Create info panel
    info = f"""[bold]Path Metrics:[/bold]
Length: {bridge_result['length']} hops
Coherence: {bridge_result['coherence']:.2%}
Novelty: {bridge_result['novelty']:.2f}
Score: {bridge_result['score']:.3f}

[bold]Explanation:[/bold]
{bridge_result['explanation']}"""
    
    panel = Panel(info, title="Bridge Analysis", border_style="blue")
    
    console.print(tree)
    console.print(panel)


def bridge_command(
    concept1: str = typer.Argument(..., help="First concept"),
    concept2: str = typer.Argument(..., help="Second concept"),
    max_hops: int = typer.Option(3, "--max-hops", "-h", help="Maximum hops between concepts"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of paths to show"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export paths to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Find conceptual bridges between two ideas"""
    
    try:
        # Load agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Loading knowledge base...", total=None)
            
            # Initialize agent with saved state
            from ...config.loader import load_config
            config = load_config()
            agent = SimpleRAGGraph(config)
            
            # Try to load saved state
            if hasattr(agent, 'load_state'):
                try:
                    agent.load_state()
                    console.print("[green]✓ Loaded knowledge base[/green]")
                except:
                    console.print("[yellow]⚠ No saved knowledge base found[/yellow]")
                    console.print("[dim]Run 'spike embed' first to build knowledge base[/dim]")
                    return
        
        # Find bridges
        console.print(f"\n[bold]Finding bridges:[/bold] {concept1} ↔ {concept2}\n")
        
        bridge_finder = ConceptBridge(agent)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Searching for conceptual paths...", total=None)
            
            bridges = bridge_finder.find_bridges(
                concept1, concept2,
                max_hops=max_hops,
                top_k=top_k
            )
        
        if not bridges:
            console.print("[yellow]No bridge concepts found[/yellow]")
            console.print("\nTry:")
            console.print("- Adding more documents with 'spike embed'")
            console.print("- Increasing --max-hops")
            console.print("- Using more general concepts")
            return
        
        # Display results
        console.print(f"[green]✨ Found {len(bridges)} conceptual paths[/green]\n")
        
        for i, bridge in enumerate(bridges, 1):
            console.print(f"[bold]═══ Path #{i} ═══[/bold]")
            visualize_bridge_path(bridge)
            console.print()
        
        # Summary table
        if len(bridges) > 1:
            table = Table(title="Path Comparison")
            table.add_column("Path #", style="cyan")
            table.add_column("Length", style="yellow")
            table.add_column("Coherence", style="green")
            table.add_column("Novelty", style="magenta")
            table.add_column("Score", style="bold")
            
            for i, bridge in enumerate(bridges, 1):
                table.add_row(
                    str(i),
                    str(bridge['length']),
                    f"{bridge['coherence']:.2%}",
                    f"{bridge['novelty']:.2f}",
                    f"{bridge['score']:.3f}"
                )
            
            console.print(table)
        
        # Export if requested
        if export:
            export_data = {
                'timestamp': time.time(),
                'query': {
                    'concept1': concept1,
                    'concept2': concept2,
                    'max_hops': max_hops
                },
                'results': bridges
            }
            
            with open(export, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"\n[green]✓ Exported to {export}[/green]")
        
        # Usage tips
        if verbose:
            console.print("\n[bold]Tips:[/bold]")
            console.print("• Shorter paths are more direct but less creative")
            console.print("• Higher novelty indicates unexpected connections")
            console.print("• High coherence means smooth conceptual transitions")
            
    except Exception as e:
        logger.error(f"Bridge command failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())