"""
geDIG HotPotQA Demo - Compare geDIG vs BM25 retrieval

Usage:
    streamlit run apps/hotpotqa_demo.py
"""

import streamlit as st
import json
import random
from pathlib import Path

# Page config
st.set_page_config(
    page_title="geDIG Demo - HotPotQA",
    page_icon="🔍",
    layout="wide"
)

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample questions from HotPotQA"""
    data_path = Path("experiments/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl")
    if not data_path.exists():
        # Try relative path from different location
        data_path = Path("../experiments/hotpotqa-benchmark/data/hotpotqa_sample_100.jsonl")

    samples = []
    if data_path.exists():
        with open(data_path) as f:
            for line in f:
                samples.append(json.loads(line))
    return samples

@st.cache_data
def load_results():
    """Load experiment results for comparison"""
    results = {}
    results_dir = Path("experiments/hotpotqa-benchmark/results")
    if not results_dir.exists():
        results_dir = Path("../experiments/hotpotqa-benchmark/results")

    # Load latest summaries
    bm25_files = sorted(results_dir.glob("bm25_*_summary.json"), reverse=True)
    gedig_files = sorted(results_dir.glob("gedig_*_summary.json"), reverse=True)

    if bm25_files:
        with open(bm25_files[0]) as f:
            results['bm25'] = json.load(f)

    if gedig_files:
        with open(gedig_files[0]) as f:
            results['gedig'] = json.load(f)

    return results

# Main UI
st.title("🔍 geDIG vs BM25 - HotPotQA Comparison")

st.markdown("""
This demo compares **geDIG** (Graph Edit Distance + Information Gain) with **BM25**
for multi-hop question answering on the HotPotQA benchmark.
""")

# Sidebar - Results Summary
with st.sidebar:
    st.header("📊 Benchmark Results")

    results = load_results()

    if results:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("BM25")
            if 'bm25' in results:
                st.metric("EM", f"{results['bm25'].get('em', 0):.1%}")
                st.metric("F1", f"{results['bm25'].get('f1', 0):.1%}")
                st.metric("P50", f"{results['bm25'].get('latency_p50_ms', 0):.0f}ms")

        with col2:
            st.subheader("geDIG")
            if 'gedig' in results:
                em_diff = results['gedig'].get('em', 0) - results['bm25'].get('em', 0)
                f1_diff = results['gedig'].get('f1', 0) - results['bm25'].get('f1', 0)

                st.metric("EM", f"{results['gedig'].get('em', 0):.1%}",
                         delta=f"{em_diff:+.1%}")
                st.metric("F1", f"{results['gedig'].get('f1', 0):.1%}",
                         delta=f"{f1_diff:+.1%}")
                st.metric("P50", f"{results['gedig'].get('latency_p50_ms', 0):.0f}ms")

        st.divider()
        st.caption(f"Dataset: {results.get('gedig', {}).get('count', 'N/A')} questions")

    st.divider()
    st.header("ℹ️ About geDIG")
    st.markdown("""
    geDIG uses a unified gauge to decide **when** to accept new information:

    ```
    F = ΔEPC - λ·ΔIG
    ```

    - **ΔEPC**: Structural cost (edit-path cost)
    - **ΔIG**: Information gain (entropy + path shortening)
    - **AG**: Ambiguity Gate (explore more?)
    - **DG**: Decision Gate (accept this?)
    """)

# Main content
samples = load_sample_data()

if not samples:
    st.error("Could not load sample data. Please run from the repository root.")
    st.stop()

# Question selector
st.header("🎯 Try a Question")

col1, col2 = st.columns([3, 1])

with col1:
    # Random question button
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0

with col2:
    if st.button("🎲 Random Question"):
        st.session_state.current_idx = random.randint(0, len(samples) - 1)

# Question selector
selected_idx = st.selectbox(
    "Select a question:",
    range(len(samples)),
    index=st.session_state.current_idx,
    format_func=lambda i: f"Q{i+1}: {samples[i]['question'][:80]}..."
)

sample = samples[selected_idx]

# Display question
st.subheader("Question")
st.info(sample['question'])

# Display answer
st.subheader("Ground Truth Answer")
st.success(sample['answer'])

# Display context
st.subheader("📚 Available Context")

with st.expander("Show all context documents", expanded=False):
    for i, (title, sentences) in enumerate(sample.get('context', [])):
        st.markdown(f"**{i+1}. {title}**")
        for j, sent in enumerate(sentences):
            # Highlight supporting facts
            is_supporting = [title, j] in sample.get('supporting_facts', [])
            if is_supporting:
                st.markdown(f"🟢 {sent}")
            else:
                st.markdown(f"   {sent}")
        st.divider()

# Supporting facts
st.subheader("✅ Supporting Facts (Ground Truth)")
supporting_facts = sample.get('supporting_facts', [])
for title, sent_idx in supporting_facts:
    # Find the sentence
    for doc_title, sentences in sample.get('context', []):
        if doc_title == title and sent_idx < len(sentences):
            st.markdown(f"- **{title}** [{sent_idx}]: {sentences[sent_idx]}")
            break

# Comparison section
st.header("🔄 Method Comparison")

st.markdown("""
| Aspect | BM25 | geDIG |
|--------|------|-------|
| **Retrieval** | Top-k by term matching | Dynamic, gate-controlled |
| **Decision** | Fixed threshold | F-score based (AG/DG) |
| **Updates** | None | Graph structure evolves |
| **Strength** | Fast, simple | Adapts to ambiguity |
""")

# geDIG explanation
st.header("🧠 How geDIG Decides")

st.markdown("""
### Two-Stage Gating

1. **AG (Ambiguity Gate)** - *"Should I explore more?"*
   - Fires when local structure is ambiguous
   - Triggers additional retrieval

2. **DG (Decision Gate)** - *"Is this a good connection?"*
   - Fires when multi-hop evaluation confirms a shortcut
   - Commits the connection to the graph

### The Unified Gauge

```
F = ΔEPC_norm - λ·(ΔH_norm + γ·ΔSP_rel)
```

- **Small F** → Good update (low cost, high gain)
- **Large F** → Bad update (high cost, low gain)
""")

# Footer
st.divider()
st.caption("""
**geDIG** - A Unified Gauge Framework for Dynamic Knowledge Graphs
[GitHub](https://github.com/miyauchikazuyoshi/InsightSpike-AI) |
[Paper](https://miyauchikazuyoshi.github.io/InsightSpike-AI)
""")
