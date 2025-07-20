"""
spike graph command - Knowledge graph analytics and visualization
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict, Counter

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
import numpy as np

from ...implementations.agents.main_agent import MainAgent as SimpleRAGGraph

logger = logging.getLogger(__name__)
console = Console()


class GraphAnalyzer:
    """Analyze and visualize knowledge graph structure"""
    
    def __init__(self, agent: SimpleRAGGraph):
        self.agent = agent
        self.l2_memory = agent.l2_memory
        self.l3_reasoner = agent.l3_graph
        
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the current knowledge graph structure"""
        if not self.l3_reasoner or not hasattr(self.l3_reasoner, 'previous_graph'):
            return {
                'error': 'No graph data available',
                'nodes': 0,
                'edges': 0
            }
            
        graph = self.l3_reasoner.previous_graph
        if graph is None:
            return {
                'error': 'Graph not initialized',
                'nodes': 0,
                'edges': 0
            }
            
        # Count nodes and edges
        nodes = set()
        edges = []
        
        # Handle different graph formats
        if hasattr(graph, 'edge_index'):
            # PyTorch Geometric graph
            edge_index = graph.edge_index.numpy() if hasattr(graph.edge_index, 'numpy') else graph.edge_index
            edge_weight = graph.edge_weight.numpy() if hasattr(graph, 'edge_weight') and hasattr(graph.edge_weight, 'numpy') else None
            
            for i in range(edge_index.shape[1]):
                src, tgt = int(edge_index[0, i]), int(edge_index[1, i])
                if src < tgt:  # Avoid duplicates
                    nodes.add(src)
                    nodes.add(tgt)
                    weight = float(edge_weight[i]) if edge_weight is not None else 1.0
                    edges.append((src, tgt, weight))
            
            # Create adjacency matrix for other functions
            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else max(nodes) + 1
            graph_2d = np.zeros((num_nodes, num_nodes))
            for src, tgt, weight in edges:
                graph_2d[src][tgt] = weight
                graph_2d[tgt][src] = weight
        else:
            # Assume it's a numpy array
            if hasattr(graph, 'ndim') and graph.ndim == 1:
                # Convert 1D array to 2D adjacency matrix
                n = int(np.sqrt(len(graph)))
                graph_2d = graph.reshape(n, n)
            else:
                graph_2d = np.array(graph)
                
            for i in range(len(graph_2d)):
                for j in range(i+1, len(graph_2d)):
                    if graph_2d[i][j] > 0:
                        nodes.add(i)
                        nodes.add(j)
                        edges.append((i, j, float(graph_2d[i][j])))
        
        # Calculate graph metrics
        num_nodes = len(nodes)
        num_edges = len(edges)
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Find most connected nodes
        node_degrees = Counter()
        for i, j, _ in edges:
            node_degrees[i] += 1
            node_degrees[j] += 1
            
        # Calculate centrality metrics
        centrality = self._calculate_centrality(graph, list(nodes))
        
        # Detect communities
        communities = self._detect_communities(edges, nodes)
        
        # Store graph_2d for later use
        self._graph_2d = graph_2d
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'density': density,
            'average_degree': (2 * num_edges) / num_nodes if num_nodes > 0 else 0,
            'most_connected': node_degrees.most_common(5),
            'centrality': centrality,
            'communities': communities,
            'edge_weights': [w for _, _, w in edges]
        }
    
    def _calculate_centrality(self, graph: np.ndarray, nodes: List[int]) -> Dict[str, List[Tuple[int, float]]]:
        """Calculate various centrality measures"""
        centrality = {
            'degree': [],
            'closeness': [],
            'betweenness': []
        }
        
        # Degree centrality
        for node in nodes:
            # Use graph_2d for consistency
            graph_2d = graph if graph.ndim == 2 else graph.reshape(int(np.sqrt(len(graph))), -1)
            degree = sum(1 for i in range(len(graph_2d)) if graph_2d[node][i] > 0 and i != node)
            centrality['degree'].append((node, degree))
        
        centrality['degree'].sort(key=lambda x: x[1], reverse=True)
        
        # Simplified closeness centrality
        for node in nodes:
            distances = self._bfs_distances(graph_2d, node)
            if distances:
                avg_distance = sum(distances.values()) / len(distances)
                closeness = 1 / avg_distance if avg_distance > 0 else 0
                centrality['closeness'].append((node, closeness))
        
        centrality['closeness'].sort(key=lambda x: x[1], reverse=True)
        
        return centrality
    
    def _bfs_distances(self, graph: np.ndarray, start: int) -> Dict[int, int]:
        """BFS to find distances from start node"""
        distances = {}
        queue = [(start, 0)]
        visited = {start}
        
        while queue:
            node, dist = queue.pop(0)
            
            for i in range(len(graph)):
                if i not in visited and graph[node][i] > 0:
                    visited.add(i)
                    distances[i] = dist + 1
                    queue.append((i, dist + 1))
        
        return distances
    
    def _detect_communities(self, edges: List[Tuple[int, int, float]], nodes: set) -> List[List[int]]:
        """Simple community detection using connected components"""
        # Build adjacency list
        adj = defaultdict(set)
        for i, j, _ in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Find connected components
        communities = []
        visited = set()
        
        for node in nodes:
            if node not in visited:
                community = []
                queue = [node]
                
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        community.append(current)
                        queue.extend(adj[current] - visited)
                
                if community:
                    communities.append(sorted(community))
        
        return sorted(communities, key=len, reverse=True)
    
    def generate_insights_network(self) -> Dict[str, Any]:
        """Generate insights about the knowledge network"""
        analysis = self.analyze_graph_structure()
        
        if 'error' in analysis:
            return analysis
        
        insights = []
        
        # Density insight
        density = analysis['density']
        if density < 0.1:
            insights.append({
                'type': 'sparse_graph',
                'message': 'Knowledge graph is sparse - consider adding more connections',
                'severity': 'warning'
            })
        elif density > 0.5:
            insights.append({
                'type': 'dense_graph',
                'message': 'Knowledge graph is highly connected - good knowledge integration',
                'severity': 'info'
            })
        
        # Hub nodes
        if analysis['most_connected']:
            top_node = analysis['most_connected'][0]
            if top_node[1] > analysis['average_degree'] * 2:
                insights.append({
                    'type': 'hub_detected',
                    'message': f'Node {top_node[0]} is a major hub with {top_node[1]} connections',
                    'severity': 'info'
                })
        
        # Community structure
        communities = analysis['communities']
        if len(communities) > 1:
            insights.append({
                'type': 'multiple_communities',
                'message': f'Found {len(communities)} separate knowledge clusters',
                'severity': 'info',
                'details': f'Sizes: {[len(c) for c in communities[:3]]}'
            })
        
        return {
            'analysis': analysis,
            'insights': insights
        }
    
    def export_for_visualization(self, output_path: Path) -> Dict[str, Any]:
        """Export graph data for external visualization tools"""
        if not self.l3_reasoner or not hasattr(self.l3_reasoner, 'previous_graph'):
            return {'error': 'No graph data available'}
            
        graph = self.l3_reasoner.previous_graph
        if graph is None:
            return {'error': 'Graph not initialized'}
        
        # Create node and edge lists
        nodes = []
        edges = []
        
        # Handle different graph formats
        if hasattr(graph, 'edge_index'):
            # PyTorch Geometric format
            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else 5
        else:
            # Numpy array format
            num_nodes = len(graph) if hasattr(graph, '__len__') else 0
        
        # Map node indices to episode content
        for i in range(min(num_nodes, len(self.l2_memory.episodes))):
            episode = self.l2_memory.episodes[i]
            nodes.append({
                'id': i,
                'label': episode.text[:50] + '...' if len(episode.text) > 50 else episode.text,
                'full_text': episode.text,
                'c_value': episode.c,
                'timestamp': episode.timestamp
            })
        
        # Extract edges
        if hasattr(graph, 'edge_index'):
            # PyTorch Geometric format
            edge_index = graph.edge_index.numpy() if hasattr(graph.edge_index, 'numpy') else graph.edge_index
            edge_weight = graph.edge_weight.numpy() if hasattr(graph, 'edge_weight') and hasattr(graph.edge_weight, 'numpy') else None
            
            for i in range(edge_index.shape[1]):
                src, tgt = int(edge_index[0, i]), int(edge_index[1, i])
                if src < tgt:  # Avoid duplicates
                    weight = float(edge_weight[i]) if edge_weight is not None else 1.0
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'weight': weight
                    })
        else:
            # Numpy array format
            graph_array = np.array(graph) if hasattr(graph, '__array__') else graph
            if hasattr(graph_array, 'shape') and len(graph_array.shape) == 2:
                for i in range(len(graph_array)):
                    for j in range(i+1, len(graph_array)):
                        if graph_array[i][j] > 0:
                            edges.append({
                                'source': i,
                                'target': j,
                                'weight': float(graph_array[i][j])
                    })
        
        # Prepare visualization data
        vis_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'generated_at': time.time(),
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'format': 'vis.js'
            }
        }
        
        # Generate HTML visualization
        html_content = self._generate_html_visualization(vis_data)
        
        # Save files
        json_path = output_path.with_suffix('.json')
        html_path = output_path.with_suffix('.html')
        
        with open(json_path, 'w') as f:
            json.dump(vis_data, f, indent=2)
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return {
            'json_path': str(json_path),
            'html_path': str(html_path),
            'nodes': len(nodes),
            'edges': len(edges)
        }
    
    def _generate_html_visualization(self, data: Dict[str, Any]) -> str:
        """Generate interactive HTML visualization using vis.js"""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>InsightSpike Knowledge Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        #mynetwork {
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
            background-color: white;
        }
        #info {
            margin-top: 20px;
            padding: 10px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .controls {
            margin-bottom: 10px;
        }
        button {
            margin-right: 10px;
            padding: 5px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>InsightSpike Knowledge Graph Visualization</h1>
    <div class="controls">
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="fitNetwork()">Fit to Screen</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
    </div>
    <div id="mynetwork"></div>
    <div id="info">
        <h3>Graph Statistics</h3>
        <p>Nodes: <span id="node-count">0</span></p>
        <p>Edges: <span id="edge-count">0</span></p>
        <p>Selected Node: <span id="selected-node">None</span></p>
    </div>
    
    <script type="text/javascript">
        // Parse the data
        var graphData = """ + json.dumps(data) + """;
        
        // Create nodes and edges
        var nodes = new vis.DataSet(graphData.nodes.map(function(node) {
            return {
                id: node.id,
                label: node.label,
                title: node.full_text,
                value: node.c_value * 20,
                color: {
                    background: getNodeColor(node.c_value),
                    border: '#2B7CE9'
                }
            };
        }));
        
        var edges = new vis.DataSet(graphData.edges.map(function(edge) {
            return {
                from: edge.source,
                to: edge.target,
                value: edge.weight,
                title: 'Weight: ' + edge.weight.toFixed(3)
            };
        }));
        
        // Create network
        var container = document.getElementById('mynetwork');
        var data = {
            nodes: nodes,
            edges: edges
        };
        
        var options = {
            nodes: {
                shape: 'dot',
                font: {
                    size: 12
                },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 2,
                shadow: true,
                smooth: {
                    type: 'continuous'
                }
            },
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                },
                stabilization: {
                    iterations: 1000
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };
        
        var network = new vis.Network(container, data, options);
        
        // Helper functions
        function getNodeColor(cValue) {
            var hue = (1 - cValue) * 240;
            return 'hsl(' + hue + ', 70%, 70%)';
        }
        
        function resetZoom() {
            network.fit();
        }
        
        function fitNetwork() {
            network.fit();
        }
        
        var physicsEnabled = true;
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({physics: {enabled: physicsEnabled}});
        }
        
        // Event handlers
        network.on("selectNode", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                document.getElementById('selected-node').textContent = node.label;
            }
        });
        
        network.on("deselectNode", function(params) {
            document.getElementById('selected-node').textContent = 'None';
        });
        
        // Update statistics
        document.getElementById('node-count').textContent = graphData.nodes.length;
        document.getElementById('edge-count').textContent = graphData.edges.length;
        
        // Stabilize and fit
        network.once('stabilizationIterationsDone', function() {
            network.setOptions({physics: false});
            network.fit();
        });
    </script>
</body>
</html>"""
        return html_template


def visualize_graph_metrics(analysis: Dict[str, Any]) -> None:
    """Display graph metrics in a beautiful format"""
    # Basic metrics panel
    metrics_text = f"""[bold]Graph Overview:[/bold]
Nodes: {analysis['nodes']}
Edges: {analysis['edges']}
Density: {analysis['density']:.2%}
Average Degree: {analysis['average_degree']:.2f}"""
    
    panel = Panel(metrics_text, title="Graph Metrics", border_style="blue")
    console.print(panel)
    
    # Most connected nodes table
    if analysis['most_connected']:
        table = Table(title="Most Connected Nodes")
        table.add_column("Node ID", style="cyan")
        table.add_column("Connections", style="yellow")
        table.add_column("Type", style="green")
        
        for node_id, degree in analysis['most_connected']:
            node_type = "Hub" if degree > analysis['average_degree'] * 2 else "Normal"
            table.add_row(str(node_id), str(degree), node_type)
        
        console.print(table)
    
    # Communities
    if 'communities' in analysis and analysis['communities']:
        console.print("\n[bold]Knowledge Communities:[/bold]")
        for i, community in enumerate(analysis['communities'][:3], 1):
            console.print(f"  Community {i}: {len(community)} nodes")
    
    # Edge weight distribution
    if 'edge_weights' in analysis and analysis['edge_weights']:
        weights = analysis['edge_weights']
        console.print(f"\n[bold]Edge Weight Statistics:[/bold]")
        console.print(f"  Min: {min(weights):.3f}")
        console.print(f"  Max: {max(weights):.3f}")
        console.print(f"  Mean: {np.mean(weights):.3f}")


def graph_command():
    """Graph subcommand group"""
    pass


def analyze_command(
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export analysis to JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis")
):
    """Analyze knowledge graph structure and metrics"""
    
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
                    return
        
        # Analyze graph
        console.print("\n[bold]Analyzing Knowledge Graph[/bold]\n")
        
        analyzer = GraphAnalyzer(agent)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Analyzing graph structure...", total=None)
            
            result = analyzer.generate_insights_network()
        
        if 'error' in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return
        
        # Display analysis
        analysis = result['analysis']
        visualize_graph_metrics(analysis)
        
        # Display insights
        if result['insights']:
            console.print("\n[bold]Graph Insights:[/bold]")
            for insight in result['insights']:
                icon = "⚠️" if insight['severity'] == 'warning' else "ℹ️"
                console.print(f"{icon} {insight['message']}")
                if 'details' in insight:
                    console.print(f"   [dim]{insight['details']}[/dim]")
        
        # Export if requested
        if export:
            export_data = {
                'timestamp': time.time(),
                'analysis': analysis,
                'insights': result['insights']
            }
            
            with open(export, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"\n[green]✓ Analysis exported to {export}[/green]")
        
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


def visualize_command(
    output: Path = typer.Argument(..., help="Output path for visualization files"),
    format: str = typer.Option("html", "--format", "-f", help="Output format: html, json"),
    insights_only: bool = typer.Option(False, "--insights-only", help="Only show nodes with insights"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Generate interactive visualization of the knowledge graph"""
    
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
                    return
        
        # Generate visualization
        console.print("\n[bold]Generating Knowledge Graph Visualization[/bold]\n")
        
        analyzer = GraphAnalyzer(agent)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Creating visualization...", total=None)
            
            result = analyzer.export_for_visualization(output)
        
        if 'error' in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return
        
        # Display results
        console.print("[green]✓ Visualization created successfully![/green]")
        console.print(f"\n[bold]Output Files:[/bold]")
        console.print(f"  📊 JSON data: {result['json_path']}")
        console.print(f"  🌐 HTML viewer: {result['html_path']}")
        console.print(f"\n[bold]Graph Size:[/bold]")
        console.print(f"  Nodes: {result['nodes']}")
        console.print(f"  Edges: {result['edges']}")
        console.print(f"\n[dim]Open the HTML file in a web browser to view the interactive graph[/dim]")
        
    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())