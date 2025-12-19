import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# --- Setup & Config ---
st.set_page_config(page_title="InsightSpike Sandbox", page_icon="🧠", layout="wide")

st.title("🧠 InsightSpike: The 'Aha!' Moment Playground")
st.markdown("""
This playground lets you experiment with **geDIG** (Generalized Differential Information Gain).
It decides **when** a Knowledge Graph should accept a new connection (an "Insight").

**Formula:** $F = \\Delta EPC_{norm} - \\lambda \\cdot \\Delta IG$
""")

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Parameters")

lambda_val = st.sidebar.slider(
    "Lambda (λ) - The 'Skepticism' Factor",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="High Lambda = Conservative (Hard to impress).\nLow Lambda = Open-minded (Easily accepts updates)."
)

st.sidebar.markdown("---")
st.sidebar.header("Scenario")
scenario = st.sidebar.radio(
    "Choose a topology:",
    ["The Shortcut (Classic)", "The Noise (Bad Edge)", "The Bridge (Community Merge)"]
)

# --- Graph Logic ---

def get_graphs(scenario_name):
    G_before = nx.Graph()
    G_after = nx.Graph()
    
    if scenario_name == "The Shortcut (Classic)":
        # A long chain vs a direct shortcut
        # Before: A-B-C-D-E
        edges_b = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        G_before.add_edges_from(edges_b)
        
        # After: Add A-E directly
        G_after.add_edges_from(edges_b)
        G_after.add_edge("A", "E")
        
        desc = "We found a direct path (A-E) that skips 3 hops."
        delta_epc = 0.2  # Mock cost
        delta_ig = 0.8   # High value (shortcut)

    elif scenario_name == "The Noise (Bad Edge)":
        # A triangle
        edges_b = [("A", "B"), ("B", "C"), ("C", "A")]
        G_before.add_edges_from(edges_b)
        
        # Add a random leaf
        G_after.add_edges_from(edges_b)
        G_after.add_edge("C", "D")
        
        desc = "Adding a leaf node (D). Does not improve overall connectivity much."
        delta_epc = 0.1  # Low cost
        delta_ig = 0.05  # Very low value
        
    else: # The Bridge
        # Two clusters
        edges_b = [("A","B"), ("B","C"), ("C","A"), ("D","E"), ("E","F"), ("F","D")]
        G_before.add_edges_from(edges_b)
        
        # Connect them
        G_after.add_edges_from(edges_b)
        G_after.add_edge("C", "D")
        
        desc = "Connecting two separate islands of knowledge."
        delta_epc = 0.3  # Moderate cost
        delta_ig = 0.9   # Huge value (global connectivity)

    return G_before, G_after, desc, delta_epc, delta_ig

G_b, G_a, description, d_epc, d_ig = get_graphs(scenario)

# --- Calculation ---
F = d_epc - (lambda_val * d_ig)
is_spike = F < 0

# --- Visualization ---

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Current Graph")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    pos1 = nx.spring_layout(G_b, seed=42)
    nx.draw(G_b, pos1, with_labels=True, node_color='lightblue', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("The Decision Gauge (F)")
    
    # Gauge Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Cost (ΔEPC)", f"{d_epc:.2f}")
    m2.metric("Gain (ΔIG)", f"{d_ig:.2f}")
    m3.metric("Result (F)", f"{F:.2f}", delta="-Spike!" if is_spike else "Reject", delta_color="inverse")
    
    st.info(description)
    
    # Progress bar visualization of the equation
    st.markdown("### Balance")
    cost_pct = min(1.0, d_epc)
    gain_scaled = min(1.0, lambda_val * d_ig)
    
    st.text(f"Structure Cost: {'█' * int(cost_pct * 20)} ({d_epc:.2f})")
    st.text(f"Info Gain     : {'▒' * int(gain_scaled * 20)} ({lambda_val * d_ig:.2f})")
    
    if is_spike:
        st.success(f"**ACCEPTED!** The gain outweighs the cost (F < 0).")
    else:
        st.error(f"**REJECTED.** Not enough value to justify the cost (F >= 0).")

with col3:
    st.subheader("Proposed Update")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    # Use consistent layout if nodes overlap
    pos2 = nx.spring_layout(G_a, seed=42) 
    # Highlight new edges
    new_edges = [e for e in G_a.edges() if e not in G_b.edges()]
    
    nx.draw(G_a, pos2, with_labels=True, node_color='lightgreen', ax=ax2)
    nx.draw_networkx_edges(G_a, pos2, edgelist=new_edges, edge_color='r', width=2, ax=ax2)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("*To run this locally: `pip install streamlit matplotlib networkx && streamlit run examples/playground.py`*")
