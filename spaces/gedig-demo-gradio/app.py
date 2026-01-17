"""
geDIG Demo - Hugging Face Spaces (Gradio version)
A unified gauge for deciding when to accept new information.
With Knowledge Graph Visualization!
"""

import gradio as gr
import random
import plotly.graph_objects as go
import networkx as nx
import math

# Sample HotPotQA questions with knowledge graph structure
SAMPLE_QUESTIONS = [
    {
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "context": [
            ["Arthur's Magazine", ["Arthur's Magazine was an American literary magazine published in the 19th century.", "It was founded in 1844."]],
            ["First for Women", ["First for Women is a women's magazine published by Bauer Media Group.", "It was launched in 1989."]]
        ],
        "kg_before": {
            "nodes": ["Question", "Arthur's Magazine", "First for Women", "Magazine", "Founded"],
            "edges": [
                ("Question", "Arthur's Magazine", "asks about"),
                ("Question", "First for Women", "asks about"),
                ("Arthur's Magazine", "Magazine", "is a"),
                ("First for Women", "Magazine", "is a"),
            ]
        },
        "kg_after": {
            "nodes": ["Question", "Arthur's Magazine", "First for Women", "Magazine", "Founded", "1844", "1989", "Answer"],
            "edges": [
                ("Question", "Arthur's Magazine", "asks about"),
                ("Question", "First for Women", "asks about"),
                ("Arthur's Magazine", "Magazine", "is a"),
                ("First for Women", "Magazine", "is a"),
                ("Arthur's Magazine", "1844", "founded"),
                ("First for Women", "1989", "founded"),
                ("1844", "1989", "before"),
                ("Arthur's Magazine", "Answer", "is answer"),
            ],
            "new_nodes": ["1844", "1989", "Answer"],
            "new_edges": [("Arthur's Magazine", "1844", "founded"), ("First for Women", "1989", "founded"), ("1844", "1989", "before"), ("Arthur's Magazine", "Answer", "is answer")],
            "shortcut": ("Question", "Answer")
        }
    },
    {
        "question": "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
        "answer": "Yes",
        "context": [
            ["Pavel Urysohn", ["Pavel Samuilovich Urysohn was a Soviet mathematician.", "He is best known for his contributions to topology."]],
            ["Leonid Levin", ["Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.", "He is known for his work in computational complexity theory."]]
        ],
        "kg_before": {
            "nodes": ["Question", "Pavel Urysohn", "Leonid Levin", "Person"],
            "edges": [
                ("Question", "Pavel Urysohn", "asks about"),
                ("Question", "Leonid Levin", "asks about"),
                ("Pavel Urysohn", "Person", "is a"),
                ("Leonid Levin", "Person", "is a"),
            ]
        },
        "kg_after": {
            "nodes": ["Question", "Pavel Urysohn", "Leonid Levin", "Person", "Mathematician", "Answer"],
            "edges": [
                ("Question", "Pavel Urysohn", "asks about"),
                ("Question", "Leonid Levin", "asks about"),
                ("Pavel Urysohn", "Person", "is a"),
                ("Leonid Levin", "Person", "is a"),
                ("Pavel Urysohn", "Mathematician", "profession"),
                ("Leonid Levin", "Mathematician", "profession"),
                ("Mathematician", "Answer", "same type"),
            ],
            "new_nodes": ["Mathematician", "Answer"],
            "new_edges": [("Pavel Urysohn", "Mathematician", "profession"), ("Leonid Levin", "Mathematician", "profession"), ("Mathematician", "Answer", "same type")],
            "shortcut": ("Question", "Answer")
        }
    },
    {
        "question": "What government position was held by the woman who portrayed Edith Bunker?",
        "answer": "United States Ambassador to the United Nations",
        "context": [
            ["Jean Stapleton", ["Jean Stapleton was an American actress.", "She portrayed Edith Bunker on All in the Family."]],
            ["Shirley Temple", ["Shirley Temple Black was an American actress and diplomat.", "She served as United States Ambassador to the United Nations."]]
        ],
        "kg_before": {
            "nodes": ["Question", "Edith Bunker", "Government Position", "Woman"],
            "edges": [
                ("Question", "Woman", "asks about"),
                ("Question", "Government Position", "asks about"),
                ("Woman", "Edith Bunker", "portrayed"),
            ]
        },
        "kg_after": {
            "nodes": ["Question", "Edith Bunker", "Government Position", "Woman", "Jean Stapleton", "UN Ambassador", "Answer"],
            "edges": [
                ("Question", "Woman", "asks about"),
                ("Question", "Government Position", "asks about"),
                ("Woman", "Edith Bunker", "portrayed"),
                ("Jean Stapleton", "Edith Bunker", "portrayed"),
                ("Jean Stapleton", "Woman", "is a"),
                ("Jean Stapleton", "UN Ambassador", "position"),
                ("UN Ambassador", "Answer", "is answer"),
            ],
            "new_nodes": ["Jean Stapleton", "UN Ambassador", "Answer"],
            "new_edges": [("Jean Stapleton", "Edith Bunker", "portrayed"), ("Jean Stapleton", "Woman", "is a"), ("Jean Stapleton", "UN Ambassador", "position"), ("UN Ambassador", "Answer", "is answer")],
            "shortcut": ("Question", "Answer")
        }
    },
    {
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "It was constructed from 1887 to 1889."]],
            ["France", ["France is a country in Western Europe.", "Paris is the capital and most populous city of France."]]
        ],
        "kg_before": {
            "nodes": ["Question", "Eiffel Tower", "Capital", "Country"],
            "edges": [
                ("Question", "Capital", "asks about"),
                ("Question", "Eiffel Tower", "mentions"),
                ("Eiffel Tower", "Country", "located in"),
            ]
        },
        "kg_after": {
            "nodes": ["Question", "Eiffel Tower", "Capital", "Country", "France", "Paris", "Answer"],
            "edges": [
                ("Question", "Capital", "asks about"),
                ("Question", "Eiffel Tower", "mentions"),
                ("Eiffel Tower", "Country", "located in"),
                ("Eiffel Tower", "France", "located in"),
                ("France", "Country", "is a"),
                ("France", "Paris", "capital"),
                ("Paris", "Capital", "is a"),
                ("Paris", "Answer", "is answer"),
            ],
            "new_nodes": ["France", "Paris", "Answer"],
            "new_edges": [("Eiffel Tower", "France", "located in"), ("France", "Country", "is a"), ("France", "Paris", "capital"), ("Paris", "Capital", "is a"), ("Paris", "Answer", "is answer")],
            "shortcut": ("Question", "Answer")
        }
    },
    {
        "question": "Which film has the director who was born earlier, El Dorado or Le notti bianche?",
        "answer": "Le notti bianche",
        "context": [
            ["El Dorado (1966 film)", ["El Dorado is a 1966 American Western film.", "It was directed by Howard Hawks, born in 1896."]],
            ["Le notti bianche", ["Le notti bianche is a 1957 Italian film.", "It was directed by Luchino Visconti, born in 1906."]]
        ],
        "kg_before": {
            "nodes": ["Question", "El Dorado", "Le notti bianche", "Film", "Director"],
            "edges": [
                ("Question", "El Dorado", "asks about"),
                ("Question", "Le notti bianche", "asks about"),
                ("El Dorado", "Film", "is a"),
                ("Le notti bianche", "Film", "is a"),
            ]
        },
        "kg_after": {
            "nodes": ["Question", "El Dorado", "Le notti bianche", "Film", "Director", "Howard Hawks", "Luchino Visconti", "1896", "1906", "Answer"],
            "edges": [
                ("Question", "El Dorado", "asks about"),
                ("Question", "Le notti bianche", "asks about"),
                ("El Dorado", "Film", "is a"),
                ("Le notti bianche", "Film", "is a"),
                ("El Dorado", "Howard Hawks", "directed by"),
                ("Le notti bianche", "Luchino Visconti", "directed by"),
                ("Howard Hawks", "1896", "born"),
                ("Luchino Visconti", "1906", "born"),
                ("1896", "1906", "before"),
                ("Le notti bianche", "Answer", "is answer"),
            ],
            "new_nodes": ["Howard Hawks", "Luchino Visconti", "1896", "1906", "Answer"],
            "new_edges": [("El Dorado", "Howard Hawks", "directed by"), ("Le notti bianche", "Luchino Visconti", "directed by"), ("Howard Hawks", "1896", "born"), ("Luchino Visconti", "1906", "born"), ("1896", "1906", "before"), ("Le notti bianche", "Answer", "is answer")],
            "shortcut": ("Question", "Answer")
        }
    }
]

BENCHMARK_RESULTS = {
    "bm25": {"em": 0.366, "f1": 0.523, "latency": 820},
    "gedig": {"em": 0.375, "f1": 0.538, "latency": 873}
}


def create_kg_figure(kg_data, title, show_new=False):
    """Create a plotly figure for knowledge graph visualization."""
    G = nx.DiGraph()

    # Add nodes
    for node in kg_data["nodes"]:
        G.add_node(node)

    # Add edges
    for src, dst, label in kg_data["edges"]:
        G.add_edge(src, dst, label=label)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge traces
    edge_traces = []
    edge_labels = []

    new_edges = kg_data.get("new_edges", []) if show_new else []
    new_edge_set = set((e[0], e[1]) for e in new_edges)

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        is_new = (edge[0], edge[1]) in new_edge_set

        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=3 if is_new else 2,
                color='#00CC66' if is_new else '#888'
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

        # Edge label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_labels.append({
            'x': mid_x,
            'y': mid_y,
            'text': edge[2].get('label', ''),
            'is_new': is_new
        })

    # Create node trace
    new_nodes = set(kg_data.get("new_nodes", [])) if show_new else set()

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Color coding
        if node == "Question":
            node_colors.append('#FF6B6B')  # Red for question
            node_sizes.append(40)
        elif node == "Answer":
            node_colors.append('#4ECDC4')  # Teal for answer
            node_sizes.append(40)
        elif node in new_nodes:
            node_colors.append('#00CC66')  # Green for new nodes
            node_sizes.append(35)
        else:
            node_colors.append('#6C5CE7')  # Purple for existing
            node_sizes.append(30)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color='white'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Add edge labels as annotations
    annotations = []
    for label_data in edge_labels:
        annotations.append(dict(
            x=label_data['x'],
            y=label_data['y'],
            text=label_data['text'],
            showarrow=False,
            font=dict(
                size=8,
                color='#00CC66' if label_data['is_new'] else '#666'
            ),
            bgcolor='rgba(255,255,255,0.7)'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='white')),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(30,30,40,1)',
        paper_bgcolor='rgba(30,30,40,1)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        annotations=annotations
    )

    return fig


def simulate_gedig_evaluation(question: str, kg_before: dict, kg_after: dict) -> dict:
    """Simulate geDIG evaluation based on graph structure."""
    random.seed(hash(question) % 2**32)

    # Calculate actual structural metrics
    nodes_added = len(kg_after.get("new_nodes", []))
    edges_added = len(kg_after.get("new_edges", []))

    # EPC: Edit Path Cost (normalized)
    epc_norm = (nodes_added * 0.1 + edges_added * 0.05)
    epc_norm = min(epc_norm, 0.5)  # Cap at 0.5

    # Information gain from path shortening
    has_shortcut = "shortcut" in kg_after
    delta_sp = 0.3 if has_shortcut else 0.1

    # Entropy reduction (more connections = more certainty)
    delta_h = -0.1 * edges_added / max(len(kg_before["edges"]), 1)

    lambda_weight = 1.0
    gamma = 1.0

    ig_norm = -delta_h + gamma * delta_sp  # Note: negative delta_h is good
    f_score = epc_norm - lambda_weight * ig_norm

    ag_fired = f_score > 0.15
    dg_fired = f_score < 0.0

    return {
        "f_score": f_score,
        "epc_norm": epc_norm,
        "delta_h": delta_h,
        "delta_sp": delta_sp,
        "ig_norm": ig_norm,
        "ag_fired": ag_fired,
        "dg_fired": dg_fired,
        "decision": "accept" if dg_fired else ("explore" if ag_fired else "hold"),
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "has_shortcut": has_shortcut
    }


def process_question(question_idx):
    """Process selected question and return results."""
    sample = SAMPLE_QUESTIONS[question_idx]
    question = sample["question"]
    answer = sample["answer"]

    # Context display
    context_text = ""
    for title, sentences in sample["context"]:
        context_text += f"**{title}**\n"
        for sent in sentences:
            context_text += f"- {sent}\n"
        context_text += "\n"

    # Knowledge graph figures
    kg_before = sample["kg_before"]
    kg_after = sample["kg_after"]

    fig_before = create_kg_figure(kg_before, "Before: Initial Knowledge Graph", show_new=False)
    fig_after = create_kg_figure(kg_after, "After: With Retrieved Information", show_new=True)

    # geDIG evaluation
    gedig_eval = simulate_gedig_evaluation(question, kg_before, kg_after)

    # Results
    f_score = gedig_eval["f_score"]
    epc = gedig_eval["epc_norm"]
    ig = gedig_eval["ig_norm"]
    decision = gedig_eval["decision"]

    ag_status = "🔴 AG Fired — Need more exploration" if gedig_eval["ag_fired"] else "🟢 AG Quiet — Structure OK"
    dg_status = "🟢 DG Fired — Shortcut confirmed!" if gedig_eval["dg_fired"] else "🟡 DG Quiet — Not yet convinced"

    # Structural change summary
    change_summary = f"""
### Structural Changes
- **Nodes added:** {gedig_eval['nodes_added']} (🟢 green)
- **Edges added:** {gedig_eval['edges_added']} (🟢 green lines)
- **Shortcut created:** {'Yes! Q→A path shortened' if gedig_eval['has_shortcut'] else 'No'}
"""

    # Formula breakdown
    formula_md = f"""
### F = ΔEPC - λ × (ΔH + γ × ΔSP)

| Component | Value | Meaning |
|-----------|-------|---------|
| ΔEPC (Structural Cost) | {epc:.3f} | Cost of adding nodes/edges |
| ΔH (Entropy Change) | {gedig_eval['delta_h']:.3f} | Uncertainty reduction |
| ΔSP (Path Shortening) | {gedig_eval['delta_sp']:.3f} | Q→A distance reduced |
| **Information Gain** | **{ig:.3f}** | Total benefit |
| **F-score** | **{f_score:.3f}** | Lower = Better update |
| **Decision** | **{decision.upper()}** | |
"""

    # Benchmark comparison
    benchmark_md = f"""
### HotPotQA Results (7,405 questions)

| Method | EM | F1 |
|--------|-----|-----|
| BM25 | 36.6% | 52.3% |
| **geDIG** | **37.5%** | **53.8%** |
"""

    return (
        question,
        answer,
        context_text,
        fig_before,
        fig_after,
        f"{f_score:.3f}",
        f"{epc:.3f}",
        f"{ig:.3f}",
        decision.upper(),
        ag_status,
        dg_status,
        change_summary,
        formula_md,
        benchmark_md
    )


# Create Gradio interface
with gr.Blocks(title="geDIG Demo", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown("""
    # 🧠 geDIG Demo — Knowledge Graph Visualization

    **geDIG** (Graph Edit Distance + Information Gain) decides **when** to accept new information by balancing:

    - **Structural Cost** (ΔEPC): How much does the graph change?
    - **Information Gain** (ΔH + ΔSP): How much does it help answer the question?

    ```
    F = Structural Cost − λ × Information Gain
    ```
    **Low F → Accept** | **High F → Reject**

    ---
    """)

    with gr.Row():
        question_dropdown = gr.Dropdown(
            choices=[(f"Q{i+1}: {q['question'][:50]}...", i) for i, q in enumerate(SAMPLE_QUESTIONS)],
            value=0,
            label="Select a Question",
            interactive=True,
            scale=3
        )

    with gr.Row():
        question_display = gr.Textbox(label="Question", interactive=False, scale=2)
        answer_display = gr.Textbox(label="Ground Truth Answer", interactive=False, scale=1)

    gr.Markdown("## 📊 Knowledge Graph: Before vs After")

    with gr.Row():
        kg_before_plot = gr.Plot(label="Before")
        kg_after_plot = gr.Plot(label="After (with new info)")

    with gr.Row():
        with gr.Column(scale=1):
            change_summary_display = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Gate Status")
            ag_status_display = gr.Textbox(label="Ambiguity Gate (AG)", interactive=False)
            dg_status_display = gr.Textbox(label="Decision Gate (DG)", interactive=False)

    gr.Markdown("## 🔬 geDIG Evaluation")

    with gr.Row():
        f_score_display = gr.Textbox(label="F-score (Lower=Better)", interactive=False)
        epc_display = gr.Textbox(label="Structural Cost", interactive=False)
        ig_display = gr.Textbox(label="Information Gain", interactive=False)
        decision_display = gr.Textbox(label="Decision", interactive=False)

    with gr.Row():
        with gr.Column():
            formula_display = gr.Markdown()
        with gr.Column():
            benchmark_display = gr.Markdown()

    with gr.Accordion("📚 Retrieved Context", open=False):
        context_display = gr.Markdown()

    gr.Markdown("""
    ---
    ### Legend
    - 🔴 **Question** node (red)
    - 🟢 **New nodes/edges** (green) — Added by retrieval
    - 🟣 **Existing nodes** (purple)
    - 🔵 **Answer** node (teal)

    ### Links
    [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) |
    [Paper](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/) |
    [5-min Guide](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes.md) |
    Contact: [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)
    """)

    # Event handlers
    question_dropdown.change(
        fn=process_question,
        inputs=[question_dropdown],
        outputs=[
            question_display,
            answer_display,
            context_display,
            kg_before_plot,
            kg_after_plot,
            f_score_display,
            epc_display,
            ig_display,
            decision_display,
            ag_status_display,
            dg_status_display,
            change_summary_display,
            formula_display,
            benchmark_display
        ]
    )

    # Load initial question
    demo.load(
        fn=process_question,
        inputs=[question_dropdown],
        outputs=[
            question_display,
            answer_display,
            context_display,
            kg_before_plot,
            kg_after_plot,
            f_score_display,
            epc_display,
            ig_display,
            decision_display,
            ag_status_display,
            dg_status_display,
            change_summary_display,
            formula_display,
            benchmark_display
        ]
    )

if __name__ == "__main__":
    demo.launch()
