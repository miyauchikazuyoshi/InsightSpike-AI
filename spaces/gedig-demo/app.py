"""
geDIG Demo - Hugging Face Spaces
A unified gauge for deciding when to accept new information.
"""

import streamlit as st
import json
import random
import math

# Page config
st.set_page_config(
    page_title="geDIG Demo",
    page_icon="🧠",
    layout="wide"
)

# Sample HotPotQA questions (embedded for Spaces)
SAMPLE_QUESTIONS = [
    {
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "context": [
            ["Arthur's Magazine", ["Arthur's Magazine was an American literary magazine published in the 19th century.", "It was founded in 1844."]],
            ["First for Women", ["First for Women is a women's magazine published by Bauer Media Group.", "It was launched in 1989."]]
        ],
        "supporting_facts": [["Arthur's Magazine", 1], ["First for Women", 1]]
    },
    {
        "question": "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
        "answer": "Yes",
        "context": [
            ["Pavel Urysohn", ["Pavel Samuilovich Urysohn was a Soviet mathematician.", "He is best known for his contributions to topology."]],
            ["Leonid Levin", ["Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.", "He is known for his work in computational complexity theory."]]
        ],
        "supporting_facts": [["Pavel Urysohn", 0], ["Leonid Levin", 0]]
    },
    {
        "question": "What government position was held by the woman who portrayed Edith Bunker?",
        "answer": "United States Ambassador to the United Nations",
        "context": [
            ["Jean Stapleton", ["Jean Stapleton was an American actress.", "She portrayed Edith Bunker on All in the Family."]],
            ["Shirley Temple", ["Shirley Temple Black was an American actress and diplomat.", "She served as United States Ambassador to the United Nations."]]
        ],
        "supporting_facts": [["Jean Stapleton", 1], ["Shirley Temple", 1]]
    },
    {
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "It was constructed from 1887 to 1889."]],
            ["France", ["France is a country in Western Europe.", "Paris is the capital and most populous city of France."]]
        ],
        "supporting_facts": [["Eiffel Tower", 0], ["France", 1]]
    },
    {
        "question": "Which film has the director who was born earlier, El Dorado or Le notti bianche?",
        "answer": "Le notti bianche",
        "context": [
            ["El Dorado (1966 film)", ["El Dorado is a 1966 American Western film.", "It was directed by Howard Hawks, born in 1896."]],
            ["Le notti bianche", ["Le notti bianche is a 1957 Italian film.", "It was directed by Luchino Visconti, born in 1906."]]
        ],
        "supporting_facts": [["El Dorado (1966 film)", 1], ["Le notti bianche", 1]]
    }
]

# Simulated benchmark results
BENCHMARK_RESULTS = {
    "bm25": {"em": 0.366, "f1": 0.523, "latency": 820},
    "gedig": {"em": 0.375, "f1": 0.538, "latency": 873, "ag_rate": 0.546, "dg_rate": 0.815}
}

def simulate_gedig_evaluation(question: str, context: list) -> dict:
    """Simulate geDIG evaluation for demo purposes."""
    # Simplified simulation
    random.seed(hash(question) % 2**32)

    epc_norm = random.uniform(0.1, 0.4)
    delta_h = random.uniform(-0.3, 0.1)
    delta_sp = random.uniform(0.0, 0.3)
    lambda_weight = 1.0
    gamma = 1.0

    ig_norm = delta_h + gamma * delta_sp
    f_score = epc_norm - lambda_weight * ig_norm

    ag_fired = f_score > 0.2
    dg_fired = f_score < 0.1

    return {
        "f_score": f_score,
        "epc_norm": epc_norm,
        "delta_h": delta_h,
        "delta_sp": delta_sp,
        "ig_norm": ig_norm,
        "ag_fired": ag_fired,
        "dg_fired": dg_fired,
        "decision": "accept" if dg_fired else ("explore" if ag_fired else "hold")
    }

def simulate_retrieval(question: str, context: list, method: str) -> dict:
    """Simulate retrieval for demo purposes."""
    random.seed(hash(question + method) % 2**32)

    # BM25: retrieves more, including noise
    # geDIG: retrieves selectively

    all_docs = []
    for title, sentences in context:
        for i, sent in enumerate(sentences):
            all_docs.append({"title": title, "idx": i, "text": sent})

    if method == "bm25":
        # Retrieve all with some noise
        retrieved = all_docs.copy()
        random.shuffle(retrieved)
    else:  # gedig
        # Retrieve more selectively
        retrieved = all_docs[:max(2, len(all_docs) - 1)]

    return {
        "method": method,
        "retrieved": retrieved,
        "count": len(retrieved)
    }

# Main UI
st.title("🧠 geDIG Demo")
st.markdown("""
**geDIG** (Graph Edit Distance + Information Gain) is a unified gauge for deciding
**when** to accept new information in RAG systems.

```
F = Structural Cost − λ × Information Gain
```
""")

# Sidebar
with st.sidebar:
    st.header("📊 Benchmark Results")
    st.markdown("*HotPotQA (7,405 questions)*")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("BM25 EM", f"{BENCHMARK_RESULTS['bm25']['em']:.1%}")
        st.metric("BM25 F1", f"{BENCHMARK_RESULTS['bm25']['f1']:.1%}")
    with col2:
        em_diff = BENCHMARK_RESULTS['gedig']['em'] - BENCHMARK_RESULTS['bm25']['em']
        f1_diff = BENCHMARK_RESULTS['gedig']['f1'] - BENCHMARK_RESULTS['bm25']['f1']
        st.metric("geDIG EM", f"{BENCHMARK_RESULTS['gedig']['em']:.1%}", f"{em_diff:+.1%}")
        st.metric("geDIG F1", f"{BENCHMARK_RESULTS['gedig']['f1']:.1%}", f"{f1_diff:+.1%}")

    st.divider()
    st.markdown("""
    ### How geDIG Works

    **AG (Ambiguity Gate)**
    - Fires when structure is uncertain
    - Triggers more exploration

    **DG (Decision Gate)**
    - Fires when shortcut is confirmed
    - Commits the update
    """)

    st.divider()
    st.markdown("""
    ### Links
    - [GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI)
    - [Paper](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/arxiv_v6_en/)
    - [5-min Guide](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/concepts/gedig_in_5_minutes.md)
    """)

# Question selector
st.header("🎯 Try a Question")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🎲 Random"):
        st.session_state.q_idx = random.randint(0, len(SAMPLE_QUESTIONS) - 1)

if 'q_idx' not in st.session_state:
    st.session_state.q_idx = 0

selected = st.selectbox(
    "Select a question:",
    range(len(SAMPLE_QUESTIONS)),
    index=st.session_state.q_idx,
    format_func=lambda i: f"Q{i+1}: {SAMPLE_QUESTIONS[i]['question'][:60]}..."
)

sample = SAMPLE_QUESTIONS[selected]

# Display question and answer
st.subheader("Question")
st.info(sample["question"])

st.subheader("Ground Truth")
st.success(f"**Answer:** {sample['answer']}")

# Context
with st.expander("📚 Available Context", expanded=False):
    for title, sentences in sample["context"]:
        st.markdown(f"**{title}**")
        for i, sent in enumerate(sentences):
            is_support = [title, i] in sample.get("supporting_facts", [])
            if is_support:
                st.markdown(f"✅ {sent}")
            else:
                st.markdown(f"　 {sent}")
        st.divider()

# Comparison
st.header("🔄 BM25 vs geDIG")

col1, col2 = st.columns(2)

with col1:
    st.subheader("BM25 (Baseline)")
    bm25_result = simulate_retrieval(sample["question"], sample["context"], "bm25")
    st.markdown(f"**Retrieved:** {bm25_result['count']} documents")
    st.markdown("**Method:** Top-k by term matching")

    with st.expander("Retrieved documents"):
        for doc in bm25_result["retrieved"]:
            st.markdown(f"- [{doc['title']}] {doc['text'][:80]}...")

with col2:
    st.subheader("geDIG (Proposed)")
    gedig_result = simulate_retrieval(sample["question"], sample["context"], "gedig")
    gedig_eval = simulate_gedig_evaluation(sample["question"], sample["context"])

    st.markdown(f"**Retrieved:** {gedig_result['count']} documents")
    st.markdown(f"**Decision:** `{gedig_eval['decision']}`")

    with st.expander("Retrieved documents"):
        for doc in gedig_result["retrieved"]:
            st.markdown(f"- [{doc['title']}] {doc['text'][:80]}...")

# geDIG Details
st.header("🔬 geDIG Evaluation Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("F-score", f"{gedig_eval['f_score']:.3f}")
    st.caption("Lower is better")

with col2:
    st.metric("Structural Cost (ΔEPC)", f"{gedig_eval['epc_norm']:.3f}")
    st.caption("Cost of editing graph")

with col3:
    st.metric("Information Gain (ΔIG)", f"{gedig_eval['ig_norm']:.3f}")
    st.caption("Entropy reduction + path shortening")

# Gate status
st.subheader("Gate Status")
col1, col2 = st.columns(2)

with col1:
    if gedig_eval["ag_fired"]:
        st.error("🔴 AG Fired — Need more exploration")
    else:
        st.success("🟢 AG Quiet — Local structure OK")

with col2:
    if gedig_eval["dg_fired"]:
        st.success("🟢 DG Fired — Shortcut confirmed, accept!")
    else:
        st.warning("🟡 DG Quiet — Not yet convinced")

# Formula explanation
st.header("📐 The Formula")

st.latex(r"F = \Delta\text{EPC}_{\text{norm}} - \lambda \cdot (\Delta H_{\text{norm}} + \gamma \cdot \Delta\text{SP}_{\text{rel}})")

st.markdown("""
| Symbol | Meaning | This Example |
|--------|---------|--------------|
| ΔEPC | Edit-path cost (structure) | {:.3f} |
| ΔH | Entropy difference | {:.3f} |
| ΔSP | Path shortening | {:.3f} |
| λ | Temperature (balance) | 1.0 |
| γ | SP weight | 1.0 |
| **F** | **Final score** | **{:.3f}** |
""".format(
    gedig_eval["epc_norm"],
    gedig_eval["delta_h"],
    gedig_eval["delta_sp"],
    gedig_eval["f_score"]
))

# Footer
st.divider()
st.caption("""
**geDIG** — A Unified Gauge Framework for Dynamic Knowledge Graphs
[GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) |
[Paper](https://github.com/miyauchikazuyoshi/InsightSpike-AI/blob/main/docs/paper/) |
Contact: [@kazuyoshim5436](https://twitter.com/kazuyoshim5436)
""")
