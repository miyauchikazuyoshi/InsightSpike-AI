
import sys
import os
print("DEBUG: Executable:", sys.executable)
print("DEBUG: Path:", sys.path)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add experiment src to path
sys.path.insert(0, os.path.abspath("archive/experiments/maze-navigation-enhanced/src"))

from navigation.maze_navigator import MazeNavigator
from analysis.clean_maze_run import generate_maze

def generate_paper_graph():
    # Configuration matches the paper/html report
    SIZE = 15
    SEED = 19
    STRATEGY = 'gedig'
    
    # generate maze
    maze = generate_maze(SIZE, SEED)
    
    # Init navigator
    start = (1, 1)
    goal = (SIZE-2, SIZE-2)
    nav = MazeNavigator(
        maze=maze,
        start_pos=start,
        goal_pos=goal,
        wiring_strategy=STRATEGY,
        gedig_threshold=-0.045,  # Matches HTML report
        backtrack_threshold=-0.02, # Matches HTML report
        simple_mode=False, # Need graph structure
        wiring_top_k=8,    # Matches HTML report
        enable_diameter_metrics=False, # Optimize speed
        dense_metric_interval=10 # Optimize speed
    )
    
    # Run till goal
    print(f"Running maze {SIZE}x{SIZE} seed={SEED}...")
    max_steps = 450 # Matches HTML report
    for _ in range(max_steps):
        nav.step()
        if nav.current_pos == goal:
            print("Goal reached!")
            break
            
    # Extract Graph
    G = nav.graph_manager.graph
    print("DEBUG: Nodes:", list(G.nodes(data=True))[:3])
    
    # Try to extract pos from node attributes if n is int
    pos = {}
    for n, data in G.nodes(data=True):
        if 'position' in data:
             p = data['position']
             pos[n] = (p[0], SIZE-1-p[1])
        elif 'pos' in data:
             p = data['pos']
             pos[n] = (p[0], SIZE-1-p[1])
        elif isinstance(n, tuple):
             pos[n] = (n[0], SIZE-1-n[1])
        else:
             # Fallback: maybe 'position' or no pos?
             pass
    
    if not pos:
        print("ERROR: Could not determine node positions.")
    
    # Flip Y for plotting (row 0 is top) 
    
    # Plot
    plt.figure(figsize=(6, 6))
    
    # Better: Plot walls as dark squares
    walls = np.argwhere(maze == 1)
    for w in walls:
        # w is (row, col) => (y, x)
        # x = w[1], y = w[0]
        # plot y inverted: SIZE-1-y
        r, c = w[0], w[1]
        x = c
        y = SIZE - 1 - r
        plt.fill([x-0.5, x+0.5, x+0.5, x-0.5], 
                 [y-0.5, y-0.5, y+0.5, y+0.5], 
                 color='#404040', alpha=0.3)

    # Draw Graph
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color='#3366cc', alpha=0.9, linewidths=1.0, edgecolors='white')
    # Edges
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color='#ff9900', alpha=0.8)
    
    # Highlight Start/Goal
    sx, sy = start[1], SIZE-1-start[0]
    gx, gy = goal[1], SIZE-1-goal[0]
    
    plt.scatter([sx], [sy], c='green', s=200, zorder=10, edgecolors='white', linewidth=2, label='Start')
    plt.scatter([gx], [gy], c='red', s=200, zorder=10, edgecolors='white', linewidth=2, label='Goal')
    
    #plt.title("Episodic Memory Graph (Seed 19)", fontsize=14)
    plt.axis('equal')
    plt.grid(False)
    plt.axis('off')
    
    # Save
    out_path = os.path.abspath("docs/paper/jsai2026/option_ab_merged/figs/maze_graph.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved graph to {out_path}")

if __name__ == "__main__":
    generate_paper_graph()
