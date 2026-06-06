"""geDIG F-Trajectory Demo — JSAI 2026 poster companion.

Visualises how the unified gauge F = ΔEPC − λ(ΔH + γ·ΔSP) evolves layer
by layer when a sentence flows through BERT. Pre-computed presets switch
instantly; custom input runs BERT live (~1-2s on Apple Silicon).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from lib import f_trajectory

# ---------- page setup ----------
st.set_page_config(
    page_title="geDIG · F-Trajectory",
    page_icon="📈",
    layout="wide",
)

# Dark theme tweaks to match the JSAI poster colour palette.
st.markdown(
    """
    <style>
      .stApp { background: #0f1320; color: #eef1f7; }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      h1, h2, h3 { color: #eef1f7; }
      .stSelectbox label, .stTextInput label { color: #aeb6c8; }
      .preset-note {
          color: #aeb6c8; font-size: 13px; font-style: italic;
          padding: 8px 12px; border-left: 3px solid #2f63cf; background: rgba(47,99,207,0.08);
      }
      .stat-card {
          background: #181d2e; border-radius: 10px; padding: 14px 18px; margin: 6px 0;
          border: 1px solid rgba(255,255,255,0.08);
      }
      .stat-name { color: #aeb6c8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
      .stat-value { color: #ffffff; font-size: 24px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colour palette aligned with the poster.
COLOR_F = "#e7b98c"        # geDIG amber
COLOR_EPC = "#993C1D"      # structural cost
COLOR_H = "#0F6E56"        # entropy term
COLOR_SP = "#3B6D11"       # path shortening
COLOR_CUMF = "#7aa8ff"     # cumulative F (highlight)

PRESETS_PATH = Path(__file__).parent / "presets.json"


# ---------- data loaders ----------
@st.cache_resource(show_spinner="Loading BERT (one-time, ~5s)…")
def _load_model(model_name: str, device: str):
    return f_trajectory.load_model(model_name, device=device)


@st.cache_data
def _load_presets() -> dict | None:
    if not PRESETS_PATH.exists():
        return None
    with open(PRESETS_PATH) as f:
        return json.load(f)


# ---------- charts ----------
def plot_cumulative_f(traj: dict, *, height: int = 360) -> go.Figure:
    """Cumulative F across layers — the headline phase-transition curve."""
    cum = traj["cumulative_f"]
    layers = list(range(1, len(cum) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=cum,
            mode="lines+markers",
            line=dict(color=COLOR_CUMF, width=4),
            marker=dict(size=9, color=COLOR_CUMF),
            name="cumulative F",
            hovertemplate="layer %{x}<br>Σ F = %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="<b>Cumulative F across layers</b> — phase-transition pattern",
        xaxis_title="Layer (transition n→n+1)",
        yaxis_title="Σ F  (smaller = more structured)",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_components(traj: dict, *, height: int = 360) -> go.Figure:
    """Per-layer ΔEPC, ΔH, ΔSP — what drives F at each layer."""
    n = len(traj["f_per_layer"])
    layers = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=layers, y=traj["epc_per_layer"], mode="lines+markers",
            line=dict(color=COLOR_EPC, width=2.5), name="ΔEPC (cost)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers, y=traj["delta_h_per_layer"], mode="lines+markers",
            line=dict(color=COLOR_H, width=2.5), name="ΔH (entropy)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers, y=traj["delta_sp_per_layer"], mode="lines+markers",
            line=dict(color=COLOR_SP, width=2.5), name="ΔSP (shortcut)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layers, y=traj["f_per_layer"], mode="lines+markers",
            line=dict(color=COLOR_F, width=3, dash="dot"),
            name="F = ΔEPC − λ(ΔH+γΔSP)",
        )
    )
    fig.update_layout(
        title="<b>Per-layer components</b> — F is a balance, not a sum",
        xaxis_title="Layer",
        yaxis_title="value",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=60, r=20, t=60, b=70),
    )
    return fig


# ---------- UI ----------
def _stat_card(name: str, value: str, *, sub: str = "") -> str:
    return (
        f"<div class='stat-card'>"
        f"<div class='stat-name'>{name}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-name' style='margin-top:4px'>{sub}</div>"
        f"</div>"
    )


def main() -> None:
    st.title("geDIG · F-Trajectory across Transformer layers")
    st.markdown(
        "Visualise how the unified gauge "
        "**F = ΔEPC − λ(ΔH + γ·ΔSP)** evolves as a sentence flows through BERT. "
        "Tied to the JSAI 2026 poster — same F, different domain."
    )

    presets = _load_presets()

    with st.sidebar:
        st.markdown("### Settings")
        model_name = st.selectbox(
            "Model",
            ["bert-base-uncased", "distilbert-base-uncased"],
            index=0,
            help="BERT-base default (12 layers). DistilBERT is faster (6 layers).",
        )
        device = st.selectbox(
            "Device",
            ["cpu", "mps", "cuda"],
            index=0,
            help="On Apple Silicon, pick 'mps' for ~3× speedup.",
        )
        lambda_ = st.slider("λ (information weight)", 0.1, 3.0, 1.0, 0.1)
        gamma = st.slider("γ (SP weight inside IG)", 0.0, 1.5, 0.5, 0.1)
        anchor_idx = 0  # [CLS] for encoder models
        st.caption(
            "Presets are pre-computed at λ=1.0, γ=0.5. "
            "Changing sliders affects custom input only."
        )

    tab_preset, tab_custom = st.tabs([
        "Presets (instant)",
        "Custom input (live BERT)",
    ])

    # --------------- Preset tab ---------------
    with tab_preset:
        if presets is None:
            st.warning(
                "Pre-computed `presets.json` not found. Run "
                "`python compute_presets.py --device mps` first."
            )
        else:
            categories = presets["categories"]
            cat_name = st.selectbox(
                "Category",
                list(categories.keys()),
                format_func=lambda s: {
                    "simple": "Simple statements",
                    "complex": "Complex / multi-clause",
                    "ambiguous": "Ambiguous parsing",
                    "garden_path": "Garden-path sentences",
                    "named_entity": "Named entity",
                    "question": "Questions",
                }.get(s, s),
            )
            items = categories[cat_name]
            options = {item["text"]: item for item in items}
            picked = st.selectbox("Sentence", list(options.keys()))
            item = options[picked]
            traj = item["trajectory"]

            if item.get("note"):
                st.markdown(
                    f"<div class='preset-note'>{item['note']}</div>",
                    unsafe_allow_html=True,
                )

            _render_result(traj)

    # --------------- Custom tab ---------------
    with tab_custom:
        text = st.text_area(
            "Enter a sentence",
            value="Although it was raining heavily, John decided to walk home.",
            height=80,
        )
        run = st.button("▶ Compute F-trajectory", type="primary")
        if run and text.strip():
            t0 = time.time()
            try:
                model, tokenizer = _load_model(model_name, device)
            except Exception as e:
                st.error(f"Could not load model: {e}")
                return
            with st.spinner(f"Running {model_name} forward pass…"):
                traj_obj = f_trajectory.compute(
                    model,
                    tokenizer,
                    text,
                    model_name=model_name,
                    anchor_idx=anchor_idx,
                    lambda_=lambda_,
                    gamma=gamma,
                    device=device,
                )
            elapsed = time.time() - t0
            st.success(f"Computed in {elapsed:.1f}s")
            _render_result(traj_obj.to_dict())


def _render_result(traj: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _stat_card(
            "Layers",
            str(traj["num_layers"]),
            sub=f"{traj['num_tokens']} tokens",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _stat_card("Total F", f"{traj['total_f']:.3f}"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _stat_card("Mean F / layer", f"{traj['mean_f']:.3f}"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _stat_card(
            "Monotonic?",
            "✓ yes" if traj.get("monotonic") else "no",
            sub="cumulative F never decreases",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_cumulative_f(traj), use_container_width=True)
    with right:
        st.plotly_chart(plot_components(traj), use_container_width=True)

    with st.expander("Raw values"):
        st.json(
            {
                "f_per_layer": [round(v, 4) for v in traj["f_per_layer"]],
                "cumulative_f": [round(v, 4) for v in traj["cumulative_f"]],
                "epc_per_layer": [round(v, 4) for v in traj["epc_per_layer"]],
                "delta_h_per_layer": [round(v, 4) for v in traj["delta_h_per_layer"]],
                "delta_sp_per_layer": [round(v, 4) for v in traj["delta_sp_per_layer"]],
            }
        )


if __name__ == "__main__":
    main()
