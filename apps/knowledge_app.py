
import streamlit as st
import os
import sys
from pathlib import Path
import streamlit.components.v1 as components

# Ensure src is in path just in case
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from insightspike.public import InsightAppWrapper

# --- Configuration & Styling ---
st.set_page_config(
    page_title="InsightSpike",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Feel
st.markdown("""
<style>
    /* Dark Mode Polish */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Polish */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    h1 { color: #7986CB; }
    
    /* Chat Input Fixed at Bottom */
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        z-index: 100;
    }
    
    /* Cards for Stats */
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app" not in st.session_state:
    st.session_state.app = None
if "ingest_text" not in st.session_state:
    st.session_state.ingest_text = ""

# --- Helper: Interactive Graph (PyVis) ---
def render_graph_pyvis(app, *, max_nodes: int, min_similarity: float, top_k: int, force_connect: bool):
    try:
        from pyvis.network import Network
        import networkx as nx
        import tempfile
        
        # Hypothetically extract NX graph
        # Prefer wrapper's graph builder if available
        G = None
        if hasattr(app, "get_graph_networkx"):
            G = app.get_graph_networkx(
                max_nodes=max_nodes,
                min_similarity=min_similarity,
                top_k=top_k,
                force_connect=force_connect,
            )
        if G is None and hasattr(app.agent, "memory") and hasattr(app.agent.memory.graph, "to_networkx"):
            G = app.agent.memory.graph.to_networkx()
        if G is None:
            # Fallback dummy for visualization if not yet available
            G = nx.Graph()
            G.add_edge("InsightSpike", "Knowledge")
            G.add_edge("Knowledge", "Power")
            G.add_edge("Local", "Fast")
            G.add_edge("InsightSpike", "Local")

        net = Network(height="500px", width="100%", bgcolor="#1E1E1E", font_color="white", notebook=False)
        net.from_nx(G)
        
        # Physics options for premium feel
        net.repulsion(node_distance=100, spring_length=200)
        net.set_options(
            """
            var options = {
              nodes: {
                shape: "dot",
                scaling: { min: 8, max: 24 },
                font: { size: 12 }
              },
              edges: {
                color: { color: "#888888" },
                smooth: false
              },
              physics: {
                barnesHut: { gravitationalConstant: -2000, springLength: 150 },
                stabilization: { iterations: 200 }
              }
            };
            """
        )
        
        # Save to temp file and read back
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            return tmp.name, G
    except ImportError:
        st.error("PyVis not installed. Graph visualization disabled.")
        return None, None
    except Exception as e:
        st.error(f"Graph check failed: {e}")
        return None, None

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("InsightSpike")
    st.markdown("*Your Personal Knowledge Base*")
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Connection Settings", expanded=False):
        provider = st.selectbox("Provider", ["ollama", "local", "openai", "mock"], index=0)
        model = st.text_input("Model", value="mistral" if provider == "ollama" else "gpt-3.5-turbo")
        api_base = st.text_input("API Base", value="http://localhost:11434/v1" if provider == "ollama" else "")
        api_key = st.text_input("API Key", value="ollama", type="password")

    with st.expander("Quick Guide", expanded=True):
        st.markdown(
            "1) Connect with a provider (mock works offline).\n"
            "2) Add a few facts in the Ingest tab (Load Sample helps).\n"
            "3) Open Knowledge Graph and refresh to see relationships.\n"
            "4) Ask in Chat or tap a Quick Question."
        )

    # Connect Button
    if st.session_state.app is None:
        if st.button("🚀 Connect / Start Engine", use_container_width=True):
            with st.spinner("Initializing Neural Core..."):
                try:
                    st.session_state.app = InsightAppWrapper(
                        provider=provider, 
                        model=model, 
                        api_base=api_base, 
                        api_key=api_key
                    )
                    st.success("Core Online")
                    st.rerun()
                except Exception as e:
                    st.error(f"Init Failed: {e}")
    else:
        st.success(f"🟢 Connected: {model}")
        if st.button("Disconnect / Reset", type="primary", use_container_width=True):
            st.session_state.app = None
            st.rerun()

    # Stats Dashboard (if connected)
    if st.session_state.app:
        st.markdown("### 📊 Live Stats")
        stats = st.session_state.app.get_stats()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><h3>{stats.get('nodes', 0)}</h3><p>Nodes</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>{stats.get('edges', 0)}</h3><p>Edges</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>{stats.get('episodes', 0)}</h3><p>Episodes</p></div>", unsafe_allow_html=True)

# --- Main Interface ---

if not st.session_state.app:
    # Landing Page
    st.markdown("<div style='text-align: center; margin-top: 100px;'><h1>Welcome to InsightSpike</h1><p>Connect your local LLM to start building your knowledge base.</p></div>", unsafe_allow_html=True)
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph", "📥 Ingest"])

# --- Tab 1: Chat ---
with tab1:
    st.header("") # Spacer
    def send_prompt(prompt: str) -> None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.app.ask(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")

    for msg in st.session_state.messages:
        avatar = "🧠" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    with st.expander("Quick Questions", expanded=False):
        recent = st.session_state.app.get_recent_episodes(limit=3)
        if recent:
            for idx, item in enumerate(recent):
                text = item.get("text", "")
                label = text if len(text) <= 60 else text[:57] + "..."
                if st.button(f"Ask: {label}", key=f"ask_recent_{idx}"):
                    send_prompt(f"What can you tell me about: {text}")
        else:
            st.caption("Ingest text to generate quick questions.")

    if prompt := st.chat_input("Ask about what you have taught so far..."):
        send_prompt(prompt)

# --- Tab 2: Graph ---
with tab2:
    st.header("Graph Structure")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_nodes = st.slider("Max nodes", 10, 200, 80, step=10)
    with c2:
        min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.5, step=0.05)
    with c3:
        top_k = st.slider("Top-K edges", 1, 10, 4, step=1)
    with c4:
        force_connect = st.checkbox("Always connect", value=True)
    st.caption("Tip: lower similarity adds more edges; higher makes a sparser graph.")
    if st.button("🔄 Refresh Visualization"):
        graph_html_path, graph_obj = render_graph_pyvis(
            st.session_state.app,
            max_nodes=max_nodes,
            min_similarity=min_similarity,
            top_k=top_k,
            force_connect=force_connect,
        )
        if graph_html_path:
            with open(graph_html_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=500, scrolling=True)
            if graph_obj is not None:
                st.caption(
                    f"Interactive Graph: {graph_obj.number_of_nodes()} nodes, {graph_obj.number_of_edges()} edges."
                )
            else:
                st.caption("Interactive Graph: Drag nodes to rearrange.")
    else:
        st.info("Click Refresh to render the current knowledge graph.")

# --- Tab 3: Ingest ---
with tab3:
    st.header("Teach New Concepts")
    sample_path = Path(current_dir) / "../examples/sample_knowledge.txt"
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Sample", use_container_width=True):
            if sample_path.exists():
                st.session_state.ingest_text = sample_path.read_text(encoding="utf-8")
            else:
                st.warning("Sample file not found.")
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state.ingest_text = ""
    with st.form("ingest_form"):
        text_input = st.text_area(
            "Paste knowledge here...",
            height=200,
            help="Text to add to the knowledge graph.",
            key="ingest_text",
        )
        submitted = st.form_submit_button("🧠 Learn & Memorize")
        
        if submitted:
            if text_input:
                with st.spinner("Processing & updating weights..."):
                    if st.session_state.app.learn(text_input):
                        st.success("Knowledge successfully integrated into graph!")
                    else:
                        st.error("Failed to integrate knowledge.")
            else:
                st.warning("Please provide text.")
