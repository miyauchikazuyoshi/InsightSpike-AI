
"""
Visualize Banana Test Graphs
============================

Generates comparative visualizations of attention graphs for:
1. Coherent Text (Definition of AI)
2. Incoherent Text (Word Salad)

Using NetworkX and Matplotlib to show the 'Shape of Meaning'.
"""

import sys
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from insightspike.gedig import compute_f_score

def create_graph_from_attention(attention_fused, tokens, percentile=0.95):
    """
    Creates a NetworkX graph from fused attention matrix.
    Using high percentile to show only the 'Strong Skeleton'.
    """
    seq_len = attention_fused.shape[0]
    
    # Thresholding
    threshold = torch.quantile(attention_fused, percentile)
    adj = (attention_fused > threshold).float().cpu().numpy()
    
    G = nx.Graph() # Undirected for visualization simplicity
    
    # Add nodes
    for i in range(seq_len):
        if i < len(tokens):
            G.add_node(i, label=tokens[i])
            
    # Add edges
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols):
        if r != c: # No self loops
            w = attention_fused[r, c].item()
            G.add_edge(r, c, weight=w)
            
    return G

def visualize_comparison():
    print("Generating visualizations...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    coherent_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals."
    incoherent_text = "Blue sky banana hamburger justice river cloud dance computer yesterday tomorrow purple monkey dishwasher piano ocean."
    
    texts = [("Coherent (Logic)", coherent_text), ("Incoherent (Chaos)", incoherent_text)]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for idx, (title, text) in enumerate(texts):
        # 1. Inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # 2. Fuse Attention (Average all heads, all layers)
        # Use only last layer for "Final Representation"? Or average all?
        # Let's use Last Layer to see the "Result" of processing.
        last_layer_attn = outputs.attentions[-1][0] # (Heads, Seq, Seq)
        avg_attn = last_layer_attn.mean(dim=0) # (Seq, Seq)
        
        # 3. Create Graph
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        # Filter special tokens for cleaner viz if needed, but keeping them for now
        
        G = create_graph_from_attention(avg_attn, tokens, percentile=0.90)
        
        # 4. Draw
        ax = axes[idx]
        pos = nx.spring_layout(G, k=1.5, seed=42) # k=distance
        
        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color="#e0f7fa", edgecolors="#006064")
        
        # Draw Edges
        weights = [G[u][v]['weight'] * 10 for u,v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color="#b0bec5", alpha=0.6)
        
        # Labels
        labels = nx.get_node_attributes(G, 'label')
        # Clean labels (remove ##)
        clean_labels = {k: v.replace("##", "") for k, v in labels.items()}
        nx.draw_networkx_labels(G, pos, labels=clean_labels, ax=ax, font_size=10, font_family="sans-serif")
        
        # Calc Metrics for Title
        clustering = nx.average_clustering(G)
        density = nx.density(G)
        ax.set_title(f"{title}\nClustering Coeff: {clustering:.3f} | Density: {density:.3f}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    output_path = os.path.abspath("experiments/rag_reranking/banana_viz.png")
    plt.savefig(output_path, dpi=150)
    print(f"Comparison saved to: {output_path}")

if __name__ == "__main__":
    visualize_comparison()
