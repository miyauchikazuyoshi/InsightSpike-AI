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
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
      h1, h2, h3 { color: #ffffff; }
      .stSelectbox label, .stTextInput label { color: #d3d8e4; }
      /* Streamlit markdown body — make sure default text is bright enough */
      .stMarkdown p, .stMarkdown li { color: #e9ecf3; }
      /* Higher contrast info box */
      .preset-note {
          color: #ffffff; font-size: 14px;
          padding: 10px 14px; border-left: 3px solid #6a9aff;
          background: #1b2840;
          border-radius: 0 6px 6px 0;
      }
      /* Higher contrast "what to look at" hint — solid amber border, bright text */
      .focus-hint {
          color: #ffffff; font-size: 15px; font-weight: 600;
          padding: 12px 16px; border-left: 4px solid #e7b98c;
          background: #2a2218;
          border-radius: 0 6px 6px 0;
          margin-bottom: 8px;
          line-height: 1.55;
      }
      .focus-hint b { color: #ffd28a; }
      .focus-hint .label {
          color: #e7b98c; text-transform: uppercase; letter-spacing: 0.08em;
          font-size: 11px; display: block; margin-bottom: 6px; font-weight: 700;
      }
      .chart-guide {
          color: #d3d8e4; font-size: 13px;
          padding: 8px 12px; line-height: 1.55;
          background: #1a1f30; border-radius: 6px;
          margin-top: 6px;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Colour palette aligned with the poster.
COLOR_F = "#e7b98c"        # geDIG amber (poster F highlight)
COLOR_EPC = "#f0a566"      # structural cost (poster .et.t-epc)
COLOR_H = "#7fa6ee"        # entropy term (poster .et.t-h)
COLOR_SP = "#7fd4b4"       # path shortening (poster .et.t-sp)
COLOR_CUMF = "#e7b98c"     # cumulative F (highlight)
COLOR_EXPLORE = "rgba(127,166,238,0.10)"   # background tint: explore zone
COLOR_STRUCTURE = "rgba(231,185,140,0.10)" # background tint: structure zone

PRESETS_PATH = Path(__file__).parent / "presets.json"

# Per-preset focus hints — what the visitor should look at to "get it" fast.
# IMPORTANT: F semantics in this demo are "cost − gain per layer transition".
# Smaller F (incl. negative) = each layer's information gain exceeded its
# structural cost. Larger F = the layer "paid" more than it "gained".
# We deliberately DO NOT claim "up = good" or "down = good"; the interesting
# signal is the *shape* and the *between-sentence comparison*, not the sign.
FOCUS_HINTS = {
    "simple_1": "短い平叙文の基準カーブ。**比較の出発点**として使う。他の文と並べると形の違いが見える。",
    "simple_2": "事実陳述。Simple1 と並べると、語数違いでも **層ごとの増分パターンは似る** 傾向。",
    "complex_1": "長文・節構造。**累積 F の到達値**が単純文より大きい(=各層の処理量が多い)。",
    "complex_2": "関係節挿入。**中盤(L4–L6)** で ΔH(エントロピー変化)の振れが大きい傾向。",
    "amb_1": "古典的曖昧構文。'like' の品詞解釈の不確かさが **ΔH の凸凹** に現れる仮説。",
    "amb_2": "Agent/patient 曖昧。**短文だが累積 F は単純文より大きい** = 短くても処理は重い。",
    "gp_1": "**最重要プリセット**。Garden-path。 **中盤 (L5–L7) で F の傾きが変化** する仮説的観察。再パースの瞬間？",
    "gp_2": "Garden-path その2。**短文なのに累積 F が急上昇** = 構文曖昧性が F に直接効く示唆。",
    "ne_1": "**'Apple' の意味曖昧性**(果物/会社)。固有名詞単独でも文脈による構造化が見える。",
    "ne_2": "歴史的事実 + 年号。**固有名詞 + 数値** で深層の F が大きく動く傾向。",
    "q_1": "質問文。**宣言文と F 軌跡の形が違う** 傾向。短い質問でも累積 F は大きめ。",
    "q_2": "Why/How 質問。Q1 と並べると **形の違い** が見える。",
}


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


def _all_preset_items(presets: dict) -> list[dict]:
    out: list[dict] = []
    for cat_name, items in presets["categories"].items():
        for item in items:
            out.append({**item, "category": cat_name})
    return out


# ---------- charts ----------
def plot_cumulative_f(
    traj: dict,
    *,
    height: int = 380,
    compare: dict | None = None,
    show_zones: bool = True,
) -> go.Figure:
    """Cumulative F across layers — the headline phase-transition curve.

    Optionally overlays a comparison trajectory in lighter colour so the
    visitor sees that *the same F* produces different curves for
    different sentences.
    """
    cum = traj["cumulative_f"]
    n = len(cum)
    layers = list(range(1, n + 1))

    fig = go.Figure()

    # Background shading: shallow vs deep layers. We avoid "good/bad"
    # labels — there's no inherent up/down preference here. We just mark
    # "shallow half" and "deep half" so the visitor can compare slope
    # between regions.
    if show_zones and n >= 6:
        midpoint = n // 2 + 0.5
        fig.add_vrect(
            x0=0.5, x1=midpoint,
            fillcolor=COLOR_EXPLORE, line_width=0,
            annotation_text="shallow layers", annotation_position="top left",
            annotation=dict(font=dict(color="#9fb6e6", size=11), bgcolor="rgba(0,0,0,0)"),
        )
        fig.add_vrect(
            x0=midpoint, x1=n + 0.5,
            fillcolor=COLOR_STRUCTURE, line_width=0,
            annotation_text="deep layers", annotation_position="top right",
            annotation=dict(font=dict(color="#e7b98c", size=11), bgcolor="rgba(0,0,0,0)"),
        )

    # Comparison trajectory (lighter), drawn first so it sits behind the main.
    if compare is not None:
        ccum = compare["cumulative_f"]
        clayers = list(range(1, len(ccum) + 1))
        fig.add_trace(
            go.Scatter(
                x=clayers, y=ccum, mode="lines+markers",
                line=dict(color="#5b6680", width=3, dash="dash"),
                marker=dict(size=7, color="#5b6680"),
                name=f"compare: {compare.get('text', 'other')[:40]}",
                hovertemplate="layer %{x}<br>Σ F = %{y:.3f}<extra></extra>",
            )
        )

    # Main trajectory.
    fig.add_trace(
        go.Scatter(
            x=layers, y=cum, mode="lines+markers",
            line=dict(color=COLOR_CUMF, width=5),
            marker=dict(size=10, color=COLOR_CUMF),
            name=f"this: {traj.get('text', '')[:40]}",
            hovertemplate="layer %{x}<br>Σ F = %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>累積 F (層を進むほど積み上がる)</b><br>"
                 "<span style='font-size:13px;color:#d3d8e4;font-weight:normal'>"
                 "F = 各層の (構造変更コスト − 情報利得)。"
                 "形 / 傾きの変わり目を見る。"
                 "</span>",
            x=0.02, xanchor="left",
        ),
        xaxis_title="Layer (層遷移 n → n+1)",
        yaxis_title="Σ F  (累積)",
        template="plotly_dark",
        paper_bgcolor="#0f1320",
        plot_bgcolor="#181d2e",
        height=height,
        margin=dict(l=60, r=20, t=80, b=50),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        showlegend=(compare is not None),
    )
    return fig


def plot_components(traj: dict, *, height: int = 380) -> go.Figure:
    """Per-layer ΔEPC, ΔH, ΔSP — what drives F at each layer."""
    n = len(traj["f_per_layer"])
    layers = list(range(1, n + 1))
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=layers, y=traj["epc_per_layer"],
        marker_color=COLOR_EPC, opacity=0.85,
        name="ΔEPC (structural cost)",
        hovertemplate="L%{x}: ΔEPC=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=traj["delta_h_per_layer"],
        marker_color=COLOR_H, opacity=0.85,
        name="ΔH (entropy)",
        hovertemplate="L%{x}: ΔH=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=traj["delta_sp_per_layer"],
        marker_color=COLOR_SP, opacity=0.85,
        name="ΔSP (shortcut)",
        hovertemplate="L%{x}: ΔSP=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=layers, y=traj["f_per_layer"], mode="lines+markers",
        line=dict(color=COLOR_F, width=4),
        marker=dict(size=10, color=COLOR_F, line=dict(color="#1a1a1a", width=1)),
        name="F (resulting gauge)",
        hovertemplate="L%{x}: F=%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>各層の構成要素</b><br>"
                 "<span style='font-size:13px;color:#d3d8e4;font-weight:normal'>"
                 "F は ΔEPC, ΔH, ΔSP の <b>バランス</b>。1項だけでは F を予測できない"
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


# ---------- main ----------
def main() -> None:
    st.title("geDIG · F-Trajectory across BERT layers")
    st.markdown(
        "**JSAI 2026 ポスター連動デモ**。"
        "迷路と同じ式 **F = ΔEPC − λ(ΔH + γ·ΔSP)** を、BERT の hidden state に層ごとに適用。"
        "_同じゲージが異なるドメインで動くか_ を文ごとに比較します。"
    )
    st.caption(
        "📝 解釈の注意: 迷路 / RAG では「**小さい F = 良い統合**」が判断基準でしたが、"
        "このデモは推論軌跡の **観察**(何かを採択する判断ではない)。"
        "F の累積方向に良し悪しはなく、見るべきは **文ごとの形の違い** と **構成要素 ΔEPC/ΔH/ΔSP のバランス**。"
    )

    # Inline legend so the bar/line colours match the poster.
    st.markdown(
        "<div class='legend-row'>"
        + _legend_chip(COLOR_EPC, "ΔEPC — 構造変更コスト")
        + _legend_chip(COLOR_H, "ΔH — エントロピー差")
        + _legend_chip(COLOR_SP, "ΔSP — 経路短縮")
        + _legend_chip(COLOR_F, "F = 統一ゲージ")
        + "</div>",
        unsafe_allow_html=True,
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
            "Device", ["cpu", "mps", "cuda"], index=0,
            help="On Apple Silicon, pick 'mps' for ~3× speedup.",
        )
        lambda_ = st.slider("λ (information weight)", 0.1, 3.0, 1.0, 0.1)
        gamma = st.slider("γ (SP weight inside IG)", 0.0, 1.5, 0.5, 0.1)
        st.caption(
            "Presets are pre-computed at λ=1.0, γ=0.5. "
            "Sliders affect Custom input only."
        )
        st.divider()
        show_zones = st.checkbox(
            "Show explore / structure phase tint", value=True,
            help="累積 F カーブの背景を半分で塗り分け、相転移を視覚化",
        )

    tab_view, tab_compare, tab_custom = st.tabs([
        "🔍 Inspect one preset",
        "⚖️ Compare two presets",
        "✍️ Custom input (live BERT)",
    ])

    # =================== Tab 1: single inspect ===================
    with tab_view:
        if presets is None:
            st.warning(
                "`presets.json` が見つかりません。`python compute_presets.py --device mps` を先に実行してください。"
            )
        else:
            categories = presets["categories"]
            col_cat, col_sent = st.columns([1, 2])
            with col_cat:
                cat_name = st.selectbox(
                    "カテゴリ",
                    list(categories.keys()),
                    format_func=lambda s: CATEGORY_LABELS.get(s, s),
                    key="single_cat",
                )
            items = categories[cat_name]
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

            _render_result(traj, show_zones=show_zones)

    # =================== Tab 2: compare two ===================
    with tab_compare:
        if presets is None:
            st.warning("`presets.json` がありません。")
        else:
            st.markdown(
                "**同じ式 F が文によって違うカーブを描く**ことを直接比較できます。"
                "迷路の F グラフが maze-instance ごとに違うのと同じ。"
            )
            all_items = _all_preset_items(presets)
            options = {f"[{it['category']}] {it['text']}": it for it in all_items}
            c1, c2 = st.columns(2)
            with c1:
                a_label = st.selectbox(
                    "A (主) — 太い橙線", list(options.keys()), index=0, key="cmp_a",
                )
            with c2:
                b_label = st.selectbox(
                    "B (比較) — グレー破線",
                    list(options.keys()),
                    index=min(6, len(options) - 1),  # default to a different category
                    key="cmp_b",
                )
            traj_a = options[a_label]["trajectory"]
            traj_b = options[b_label]["trajectory"]
            traj_a = {**traj_a, "text": options[a_label]["text"]}
            traj_b = {**traj_b, "text": options[b_label]["text"]}

            st.plotly_chart(
                plot_cumulative_f(traj_a, compare=traj_b, show_zones=show_zones, height=440),
                use_container_width=True,
            )

            # Compact side-by-side stats
            sa, sb = st.columns(2)
            for col, traj, label in [(sa, traj_a, "A"), (sb, traj_b, "B")]:
                with col:
                    st.markdown(f"#### {label}: _{traj['text']}_")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(
                        _stat_card("Total F", f"{traj['total_f']:.3f}"),
                        unsafe_allow_html=True,
                    )
                    cc2.markdown(
                        _stat_card("Mean F / layer", f"{traj['mean_f']:.3f}"),
                        unsafe_allow_html=True,
                    )
                    cc3.markdown(
                        _stat_card(
                            "Monotonic",
                            "✓" if traj.get("monotonic") else "no",
                            sub="累積Fが減らない",
                        ),
                        unsafe_allow_html=True,
                    )

    # =================== Tab 3: custom input ===================
    with tab_custom:
        text = st.text_area(
            "文を入力 (英語推奨)",
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
            with st.spinner(f"{model_name} forward pass…"):
                traj_obj = f_trajectory.compute(
                    model, tokenizer, text,
                    model_name=model_name, anchor_idx=0,
                    lambda_=lambda_, gamma=gamma, device=device,
                )
            elapsed = time.time() - t0
            st.success(f"Computed in {elapsed:.1f}s")
            st.markdown(
                "<div class='focus-hint'>"
                "<span class='label'>何を見るか</span>"
                "Preset と並べて比較するなら、累積 F の <b>絶対値</b> ではなく <b>形(傾き)</b> に注目。"
                "短文同士・長文同士で比べるのが公平。"
                "</div>",
                unsafe_allow_html=True,
            )
            _render_result(traj_obj.to_dict(), show_zones=show_zones)


def _render_result(traj: dict, *, show_zones: bool = True) -> None:
    """4 stat cards + 2 explanatory charts side by side, with reading guide."""
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        _stat_card(
            "Layers", str(traj["num_layers"]),
            sub=f"{traj['num_tokens']} tokens",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _stat_card("Total F", f"{traj['total_f']:.3f}", sub="= ΣF over all transitions"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _stat_card("Mean F / layer", f"{traj['mean_f']:.3f}"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _stat_card(
            "層別 F が常に≥0",
            "✓ yes" if traj.get("monotonic") else "no",
            sub="各層の F が非負(コスト≥利得)か",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_cumulative_f(traj, show_zones=show_zones),
            use_container_width=True,
        )
        st.markdown(
            "<div class='chart-guide'>"
            "👁 <b>見方</b>: F は各層の (構造変更コスト − 情報利得)。"
            "<b>累積値の上下方向に良し悪しはない</b>(F は判断基準ではなく観察量)。"
            "見るべきは<b>形 / 傾きの変わり目</b>。中盤で勾配が変わるなら、"
            "その文の難所(再パース・曖昧解消)に対応する可能性。"
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(plot_components(traj), use_container_width=True)
        st.markdown(
            "<div class='chart-guide'>"
            "👁 <b>見方</b>: 棒3本(ΔEPC/ΔH/ΔSP)は <b>独立に動く</b>。"
            "上の折れ線(F)はそれらの<b>バランス</b>。"
            "<b>1項だけでは F を予測できない</b> ことが、F を統一スカラーとして導入する根拠。"
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Raw values (詳細データ)"):
        st.json({
            "f_per_layer": [round(v, 4) for v in traj["f_per_layer"]],
            "cumulative_f": [round(v, 4) for v in traj["cumulative_f"]],
            "epc_per_layer": [round(v, 4) for v in traj["epc_per_layer"]],
            "delta_h_per_layer": [round(v, 4) for v in traj["delta_h_per_layer"]],
            "delta_sp_per_layer": [round(v, 4) for v in traj["delta_sp_per_layer"]],
        })


if __name__ == "__main__":
    main()
