"""geDIG · Attention F — JSAI 2026 paper §3 companion demo.

Reproduces the experimental setup of the JSAI 2026 paper Section 3:
attention-based F per (layer × head), real vs random baseline.

This is the canonical formula from
    src/insightspike/algorithms/gedig/attention.py
    F = ΔEPC − λ·γ·ΔSP − λ·ΔH

Defaults λ=0.5, γ=0.5, top-10% percentile threshold match the
Phase 1 score_full.json (paper's headline numbers).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import attention_f

# ---------- page setup ----------
st.set_page_config(
    page_title="geDIG · Attention F",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0f1320; color: #eef1f7; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
      h1, h2, h3 { color: #ffffff; }
      .stSelectbox label, .stTextInput label { color: #d3d8e4; }
      .stMarkdown p, .stMarkdown li { color: #e9ecf3; }
      .preset-note {
          color: #ffffff; font-size: 14px;
          padding: 10px 14px; border-left: 3px solid #6a9aff;
          background: #1b2840; border-radius: 0 6px 6px 0;
      }
      .focus-hint {
          color: #ffffff; font-size: 15px; font-weight: 600;
          padding: 12px 16px; border-left: 4px solid #e7b98c;
          background: #2a2218; border-radius: 0 6px 6px 0;
          margin-bottom: 8px; line-height: 1.55;
      }
      .focus-hint b { color: #ffd28a; }
      .focus-hint .label {
          color: #e7b98c; text-transform: uppercase; letter-spacing: 0.08em;
          font-size: 11px; display: block; margin-bottom: 6px; font-weight: 700;
      }
      .chart-guide {
          color: #d3d8e4; font-size: 13px;
          padding: 8px 12px; line-height: 1.55;
          background: #1a1f30; border-radius: 6px; margin-top: 6px;
      }
      .chart-guide b { color: #ffffff; }
      .stat-card {
          background: #1b2236; border-radius: 10px; padding: 12px 16px; margin: 4px 0;
          border: 1px solid rgba(255,255,255,0.12);
      }
      .stat-name { color: #b8c2d6; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
      .stat-value { color: #ffffff; font-size: 22px; font-weight: 700; }
      .legend-row { display: flex; gap: 18px; flex-wrap: wrap; margin: 8px 0 16px; font-size: 13px; }
      .legend-item { display: flex; align-items: center; gap: 7px; color: #d3d8e4; font-weight: 500; }
      .legend-swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,0.3); }
      .paper-match {
          background: #1d2e1f; color: #c4e6c8;
          padding: 6px 10px; border-radius: 5px; font-size: 12px;
          display: inline-block; border: 1px solid #3a5b3d;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colours
COLOR_REAL = "#e7b98c"          # real attention (poster F amber)
COLOR_RANDOM = "#7a8aa6"        # random baseline (muted)
COLOR_EPC = "#f0a566"
COLOR_H = "#7fa6ee"
COLOR_SP = "#7fd4b4"
COLOR_DELTA = "#5fb494"         # delta F (gain)

PRESETS_PATH = Path(__file__).parent / "presets.json"

# Per-preset focus hints — Section-3 phenomenon to look for.
FOCUS_HINTS = {
    "simple_1": "短い平叙文の基準。**全 12 層で Real F > Random F**。深い層ほど F が 0 に近づく傾向を確認。",
    "simple_2": "Simple1 と並べると **層別パターンが似る**。簡単な文は層構造が安定。",
    "complex_1": "長文・節構造。**Win rate ≈ 100%**(全 (層×ヘッド)で Real が Random を上回る)。",
    "complex_2": "関係節挿入。長文でも win rate は高い。",
    "amb_1": "曖昧構文。短文だが **ΔF (Real−Random)** が小さめ ＝ 構造的優位が弱い仮説。",
    "amb_2": "Agent/patient 曖昧。同上。",
    "gp_1": "Garden-path。ΔF が他の文より **小さい**(=構造性の利得が少ない)観察。",
    "gp_2": "Garden-path その2。同じ傾向。",
    "ne_1": "**'Apple' の意味曖昧性**(果物/会社)。固有名詞の構造化。",
    "ne_2": "歴史的事実 + 年号。**ΔF が大きい**(構造的利得高い)傾向。",
    "q_1": "質問文。宣言文と F 軌跡の形を比較。",
    "q_2": "Why/How 質問。",
}


@st.cache_resource(show_spinner="Loading BERT (~5s)…")
def _load_model(model_name: str, device: str):
    return attention_f.load_model(model_name, device=device)


@st.cache_data
def _load_presets() -> dict | None:
    if not PRESETS_PATH.exists():
        return None
    with open(PRESETS_PATH) as f:
        return json.load(f)


def _all_preset_items(presets: dict) -> list[dict]:
    out: list[dict] = []
    for cat_name, items in presets["categories"].items():
        for item in items:
            out.append({**item, "category": cat_name})
    return out


# ---------- charts ----------
def plot_layer_real_vs_random(traj: dict, *, height: int = 380) -> go.Figure:
    """Layer-by-layer F: Real attention vs Random baseline.

    Paper Figure 2 reproduction: in BERT, F rises from ~-0.47 (layer 0)
    to ~-0.40 (deep layers) as structure forms; the Random baseline stays
    flat near -0.51.
    """
    f_real = traj["f_per_layer"]
    f_rand = traj["f_random_per_layer"]
    layers = list(range(len(f_real)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=f_rand, mode="lines+markers",
        line=dict(color=COLOR_RANDOM, width=3, dash="dash"),
        marker=dict(size=8, color=COLOR_RANDOM),
        name="Random attention",
        hovertemplate="L%{x}: F=%{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=f_real, mode="lines+markers",
        line=dict(color=COLOR_REAL, width=5),
        marker=dict(size=11, color=COLOR_REAL,
                    line=dict(color="#1a1a1a", width=1.5)),
        name="Real attention",
        hovertemplate="L%{x}: F=%{y:+.3f}<extra></extra>",
    ))
    # Shaded delta region
    fig.add_trace(go.Scatter(
        x=layers + layers[::-1],
        y=f_real + f_rand[::-1],
        fill="toself", fillcolor="rgba(95,180,148,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="ΔF (gain)", showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Layer-by-layer F: Real vs Random</b><br>"
                 "<span style='font-size:13px;color:#d3d8e4;font-weight:normal'>"
                 "論文 Section 3 の主結果。Real は常に上、深層で 0 に近づく(=構造化)。"
                 "</span>",
            x=0.02, xanchor="left",
        ),
        xaxis_title="Layer",
        yaxis_title="F  (less negative = more structured)",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
        margin=dict(l=60, r=20, t=80, b=70),
    )
    return fig


def plot_layer_head_heatmap(traj: dict, *, height: int = 380) -> go.Figure:
    """Heatmap of F per (layer × head). Reveals head diversity within a layer."""
    z = np.array(traj["f_layer_head"]).T  # head on Y, layer on X
    L, H = z.shape[1], z.shape[0]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=list(range(L)), y=list(range(H)),
        colorscale=[
            [0.0, "#2c3a5e"],   # very negative = blue
            [0.5, "#aeb6c8"],   # mid = grey
            [1.0, "#e7b98c"],   # close to 0 = amber (structured)
        ],
        colorbar=dict(title="F"),
        hovertemplate="Layer %{x}, Head %{y}<br>F=%{z:+.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>F per (layer × head)</b><br>"
                 "<span style='font-size:13px;color:#d3d8e4;font-weight:normal'>"
                 "明るい (amber) = F が 0 に近く構造的。同じ層でもヘッドごとに違う"
                 "</span>",
            x=0.02, xanchor="left",
        ),
        xaxis_title="Layer",
        yaxis_title="Head",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


def plot_components(traj: dict, *, height: int = 380) -> go.Figure:
    """Per-layer ΔEPC, ΔH, ΔSP — what drives F."""
    n = len(traj["f_per_layer"])
    layers = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=traj["epc_per_layer"],
        marker_color=COLOR_EPC, opacity=0.85,
        name="ΔEPC (edge density)",
        hovertemplate="L%{x}: ΔEPC=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=traj["h_per_layer"],
        marker_color=COLOR_H, opacity=0.85,
        name="ΔH (entropy)",
        hovertemplate="L%{x}: ΔH=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=traj["sp_per_layer"],
        marker_color=COLOR_SP, opacity=0.85,
        name="ΔSP (path efficiency)",
        hovertemplate="L%{x}: ΔSP=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=traj["f_per_layer"], mode="lines+markers",
        line=dict(color=COLOR_REAL, width=4),
        marker=dict(size=10, color=COLOR_REAL, line=dict(color="#1a1a1a", width=1.5)),
        name="F (resulting gauge)",
        hovertemplate="L%{x}: F=%{y:+.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>F の構成要素 (Real attention)</b><br>"
                 "<span style='font-size:13px;color:#d3d8e4;font-weight:normal'>"
                 "F は 3 項のバランス。1 項だけでは F の動きを予測できない"
                 "</span>",
            x=0.02, xanchor="left",
        ),
        xaxis_title="Layer",
        yaxis_title="value",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        barmode="group",
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        margin=dict(l=60, r=20, t=80, b=70),
    )
    return fig


# ---------- UI helpers ----------
def _stat_card(name: str, value: str, *, sub: str = "") -> str:
    return (
        f"<div class='stat-card'>"
        f"<div class='stat-name'>{name}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-name' style='margin-top:4px'>{sub}</div>"
        f"</div>"
    )


def _legend_chip(color: str, label: str) -> str:
    return (
        f"<div class='legend-item'>"
        f"<span class='legend-swatch' style='background:{color}'></span>{label}"
        f"</div>"
    )


CATEGORY_LABELS = {
    "simple": "Simple statements",
    "complex": "Complex / multi-clause",
    "ambiguous": "Ambiguous parsing",
    "garden_path": "Garden-path sentences",
    "named_entity": "Named entity",
    "question": "Questions",
}


def main() -> None:
    st.title("geDIG · Attention F across BERT layers")
    st.markdown(
        "**JSAI 2026 ポスター §3 連動デモ**。"
        "BERT の Attention 行列をグラフ化して "
        "**F = ΔEPC − λ·γ·ΔSP − λ·ΔH** を計算。"
        "Real attention vs Random baseline で **実 attention の構造的優位** を見る。"
    )
    presets = _load_presets()

    st.caption(
        "📝 解釈の約束: 論文 §3 では F は **負値**で測定され、"
        "**F が 0 に近づく = 構造化が進む**(エントロピー集中、効率パス形成)。"
        "Real は Random より上(less negative)で、深層ほどさらに 0 に近づく。"
    )

    # Legend
    st.markdown(
        "<div class='legend-row'>"
        + _legend_chip(COLOR_REAL, "Real attention F")
        + _legend_chip(COLOR_RANDOM, "Random baseline F")
        + _legend_chip(COLOR_EPC, "ΔEPC (edge density)")
        + _legend_chip(COLOR_H, "ΔH (entropy)")
        + _legend_chip(COLOR_SP, "ΔSP (path efficiency)")
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Settings")
        model_name = st.selectbox(
            "Model",
            ["bert-base-uncased", "distilbert-base-uncased"],
            index=0,
        )
        device = st.selectbox(
            "Device", ["cpu", "mps", "cuda"], index=0,
            help="Apple Silicon: 'mps' for speedup.",
        )
        lambda_ = st.slider("λ", 0.1, 2.0, 0.5, 0.05,
                            help="Phase 1 paper setting = 0.5")
        gamma = st.slider("γ", 0.0, 1.5, 0.5, 0.05)
        percentile = st.slider("Top percentile threshold", 0.5, 0.99, 0.9, 0.05,
                                help="Paper: 0.9 (= top 10% attention weights)")
        st.caption(
            "Presets are pre-computed at λ=0.5, γ=0.5, percentile=0.9. "
            "Sliders affect Custom input only."
        )
        st.markdown(
            "<span class='paper-match'>📄 数値: Phase 1 (-0.43) と一致確認済み</span>",
            unsafe_allow_html=True,
        )

    tab_view, tab_compare, tab_custom = st.tabs([
        "🔍 Inspect one preset",
        "⚖️ Compare two presets",
        "✍️ Custom input (live BERT)",
    ])

    # ----- Tab 1 -----
    with tab_view:
        if presets is None:
            st.warning(
                "`presets.json` 未生成。`python compute_presets.py --device mps` 実行してください。"
            )
        else:
            cats = presets["categories"]
            col_cat, col_sent = st.columns([1, 2])
            with col_cat:
                cat_name = st.selectbox(
                    "カテゴリ",
                    list(cats.keys()),
                    format_func=lambda s: CATEGORY_LABELS.get(s, s),
                    key="single_cat",
                )
            items = cats[cat_name]
            options = {item["text"]: item for item in items}
            with col_sent:
                picked = st.selectbox("文を選択", list(options.keys()), key="single_text")
            item = options[picked]
            traj = item["trajectory"]

            hint = FOCUS_HINTS.get(item["id"], "")
            if hint:
                st.markdown(
                    f"<div class='focus-hint'>"
                    f"<span class='label'>何を見るか</span>{hint}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if item.get("note"):
                st.markdown(
                    f"<div class='preset-note'>{item['note']}</div>",
                    unsafe_allow_html=True,
                )

            _render_result(traj)

    # ----- Tab 2 -----
    with tab_compare:
        if presets is None:
            st.warning("`presets.json` がありません。")
        else:
            st.markdown(
                "**同じ式 F が文によって違うパターンを描く**。"
                "Real vs Random の差(ΔF)も文ごとに違う。"
            )
            all_items = _all_preset_items(presets)
            options = {f"[{it['category']}] {it['text']}": it for it in all_items}
            c1, c2 = st.columns(2)
            with c1:
                a_label = st.selectbox(
                    "A — 太い橙線", list(options.keys()), index=0, key="cmp_a",
                )
            with c2:
                b_label = st.selectbox(
                    "B — 比較",
                    list(options.keys()),
                    index=min(6, len(options) - 1),
                    key="cmp_b",
                )
            traj_a = options[a_label]["trajectory"]
            traj_b = options[b_label]["trajectory"]

            # Side-by-side stat cards
            sa, sb = st.columns(2)
            for col, traj, label in [(sa, traj_a, "A"), (sb, traj_b, "B")]:
                with col:
                    st.markdown(f"#### {label}: _{traj['text']}_")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(
                        _stat_card("Real F (mean)", f"{traj['f_mean_real']:+.3f}"),
                        unsafe_allow_html=True,
                    )
                    cc2.markdown(
                        _stat_card("ΔF (Real − Rand)", f"{traj['delta_f']:+.3f}"),
                        unsafe_allow_html=True,
                    )
                    cc3.markdown(
                        _stat_card("Win rate", f"{traj['win_rate']*100:.0f}%"),
                        unsafe_allow_html=True,
                    )

            # Overlay layer plot
            fig = go.Figure()
            n_a = len(traj_a["f_per_layer"])
            n_b = len(traj_b["f_per_layer"])
            fig.add_trace(go.Scatter(
                x=list(range(n_a)), y=traj_a["f_per_layer"],
                mode="lines+markers", name=f"A — {traj_a['text'][:40]}",
                line=dict(color=COLOR_REAL, width=5),
                marker=dict(size=10),
            ))
            fig.add_trace(go.Scatter(
                x=list(range(n_b)), y=traj_b["f_per_layer"],
                mode="lines+markers", name=f"B — {traj_b['text'][:40]}",
                line=dict(color="#9aa8c4", width=3, dash="dash"),
                marker=dict(size=8),
            ))
            fig.update_layout(
                title="<b>Layer-by-layer F: A vs B (Real attention)</b>",
                xaxis_title="Layer",
                yaxis_title="F",
                template="plotly_dark",
                paper_bgcolor="#0f1320", plot_bgcolor="#181d2e",
                height=420,
                legend=dict(orientation="h", y=-0.15, font=dict(size=12)),
                margin=dict(l=60, r=20, t=60, b=70),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "<div class='chart-guide'>"
                "👁 <b>見方</b>: 両方とも Real attention の F。"
                "<b>同じ BERT・同じ式</b> なのに文ごとに曲線が違う = "
                "F が文の構造に応答していることの証拠。"
                "</div>",
                unsafe_allow_html=True,
            )

    # ----- Tab 3 -----
    with tab_custom:
        text = st.text_area(
            "文を入力 (英語推奨)",
            value="Although it was raining heavily, John decided to walk home.",
            height=80,
        )
        run = st.button("▶ Compute F (Real vs Random)", type="primary")
        if run and text.strip():
            t0 = time.time()
            try:
                model, tokenizer = _load_model(model_name, device)
            except Exception as e:
                st.error(f"Could not load model: {e}")
                return
            with st.spinner(f"{model_name} forward + random baseline…"):
                traj_obj = attention_f.compute(
                    model, tokenizer, text,
                    model_name=model_name,
                    lambda_=lambda_, gamma=gamma, percentile=percentile,
                    device=device,
                )
            elapsed = time.time() - t0
            st.success(f"Computed in {elapsed:.1f}s")
            _render_result(traj_obj.to_dict())


def _render_result(traj: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        _stat_card("Layers × Heads",
                    f"{traj['num_layers']} × {traj['num_heads']}",
                    sub=f"{traj['num_tokens']} valid tokens"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _stat_card("Real F (mean)", f"{traj['f_mean_real']:+.3f}",
                    sub="paper Phase 1: -0.43"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _stat_card("Random F", f"{traj['f_mean_random']:+.3f}",
                    sub="paper: -0.52"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _stat_card("ΔF (Real − Rand)", f"{traj['delta_f']:+.3f}",
                    sub="paper: +0.08"),
        unsafe_allow_html=True,
    )
    c5.markdown(
        _stat_card("Win rate", f"{traj['win_rate']*100:.0f}%",
                    sub="paper: 90.5%"),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_layer_real_vs_random(traj), use_container_width=True)
        st.markdown(
            "<div class='chart-guide'>"
            "👁 <b>見方</b>: 橙(Real) が灰(Random)より<b>常に上</b>。"
            "深い層ほど Real は 0 に近づく(=構造化が進む)。"
            "灰は層を進んでもほぼ平坦。"
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(plot_layer_head_heatmap(traj), use_container_width=True)
        st.markdown(
            "<div class='chart-guide'>"
            "👁 <b>見方</b>: 同じ層でも<b>ヘッドによって F が違う</b>。"
            "明るい (橙) = 構造的なヘッド、暗い (青) = ランダム的なヘッド。"
            "「役割分担」が可視化される。"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### 📊 F の内訳 (Real attention)")
    st.plotly_chart(plot_components(traj), use_container_width=True)
    st.markdown(
        "<div class='chart-guide'>"
        "👁 <b>見方</b>: F は ΔEPC, ΔH, ΔSP の <b>バランス</b>。"
        "1 項だけでは F を予測できない ＝ <b>統一スカラーとして導入する根拠</b>。"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Raw values"):
        st.json({
            "f_per_layer": [round(v, 4) for v in traj["f_per_layer"]],
            "f_random_per_layer": [round(v, 4) for v in traj["f_random_per_layer"]],
            "epc_per_layer": [round(v, 4) for v in traj["epc_per_layer"]],
            "h_per_layer": [round(v, 4) for v in traj["h_per_layer"]],
            "sp_per_layer": [round(v, 4) for v in traj["sp_per_layer"]],
        })


if __name__ == "__main__":
    main()
