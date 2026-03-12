# Splatting-Attention 双対性と geDIG-Transformer への適用

**Date:** 2026-03-06
**Origin:** TSD-OCR プロジェクトでの実験的発見
**Status:** 理論メモ → 実験設計の基盤

---

## 1. 発見の経緯

TSD-OCR で幾何プリミティブ（曲率κ, 方位θ）から文字認識するパイプラインを構築した過程で、
2つの処理経路がそれぞれ Gaussian splatting と self-attention に対応していることに気づいた。

```
幾何プリミティブ (32点: κ, θ, x, y, ...)
       │
       ├──→ Gaussian Splatting → 32ch flow field → CNN (19K)   = where pathway
       │    「この点は空間的にここに寄与する」
       │
       └──→ MLP → Self-Attention → MaxPool → FC (41K)          = what pathway
            「この点はあの点と関係がある」
```

両経路を統合（C+P Fusion, ~79K）すると、CNN→Attention→Cross-Attention という
AI史の計算構造の進化が 3 桁少ないパラメータで再現される。

---

## 2. 数学的双対性

### 統一的な集約操作

```
一般化:
  output(query) = Σ_i  K(query, point_i) · V(point_i)

Splatting:   K = exp(-||x - p_i||² / 2σ²)      query = 空間位置 x
Attention:   K = softmax(q · k_i / √d)           query = 内容ベクトル q
Cross-Attn:  K = softmax(q_A · k_B / √d)         query = 別経路のトークン
```

| | Gaussian Splatting | Self-Attention |
|---|---|---|
| カーネル定義域 | **空間** (where) | **内容** (what) |
| カーネル形状 | ガウシアン（固定） | softmax（学習） |
| 出力形式 | 空間格子マップ | 点ごとの更新ベクトル |
| パラメータ | 0（解析的） | Q,K,V 射影（学習） |

**一言で: splatting は「空間に固定された attention」、attention は「空間から解放された splatting」。**

### 合成カーネル（Attention-Weighted Splatting）

```
output(x) = Σ_i  attn_weight(i) · f_i · exp(-||x - p_i||² / 2σ²)
            ~~~~~~~~~~~~~~~~~~~           ~~~~~~~~~~~~~~~~~~~~~~~~
            what-based (learned)           where-based (analytic)
```

what が「どの点が重要か」、where が「空間のどこに置くか」を決定。
3DGS の opacity 学習と構造的に同じ。

---

## 3. geDIG-Transformer への対応

### 3.1 AG/DG = where/what の双対

| TSD-OCR | geDIG-Transformer | 役割 |
|---------|-------------------|------|
| Splatting (where) | **AG** (0-hop gauge g₀) | 局所構造の解析的判定 |
| Attention (what) | **DG** (multi-hop search) | 内容的関係の学習的判定 |
| Attn-weighted splatting | **AG→DG cascade** | where で候補絞り → what で確定 |

AG は「空間的に（構造的に）近いものを見る」= where-based filtering。
DG は「内容的に関係あるものを見る」= what-based confirmation。

**TSD-OCR で実験的に確認されたこと:**
- where だけ (CNN): EMNIST 74.38%, z→W 0/10
- what だけ (Attention): EMNIST 73.79%, z→W 7/10
- **両方必要** — where は空間パターン、what は点間関係。片方では不完全。

→ geDIG でも AG（局所構造）だけでは遠方の意味的接続を見逃し、
  DG（意味探索）だけでは局所構造の異常を見逃す。**両方必要。**

### 3.2 Phase 1/Phase 2 = Splatting/Attention の時間的分離

```
Phase 1 (Awake):
  AG = where-based: 局所構造をガウシアン的に走査（高速、解析的）
  → 「このノードの近傍に異常がある」

  DG = what-based: 意味的接続を attention 的に探索（低速、学習的）
  → 「この遠方ノードと構造的類似がある」

Phase 2 (Sleep):
  頻出パス（attention で発見）を FFN に焼き付ける
  = what-based な発見を where-based な直感に変換
  = 「attention → splatting 化」
```

**Sleep consolidation = attention の splatting 化。**
multi-hop の意味的探索（what）を、1-hop の局所応答（where）に蒸留する。
これは TSD-OCR の C22 Pre-B（Sleep NREM consolidation）と同じ原理。

### 3.3 Curl 検出との接続

```
Attention field の curl:
  ∇ × A = 非対称成分 = 回転的情報フロー

Splatting field の curl:
  隣接プリミティブの θ の変化 = dθ/ds = κ（曲率）
```

**曲率 κ は splatting field の curl そのもの。**

TSD-OCR で κ が最も重要な特徴量であるのは、
curl（=局所的な構造変化）が認識にとって最も情報量が高いから。

geDIG-Transformer でも attention field の curl が
「思考の渦 = 洞察の芯」を示す。同じ原理。

---

## 4. 実験設計への示唆

### 4.1 Dynamic Transformer の AG を splatting 的に設計

現在の AG: ベクトル類似度のみで高速判定

**提案**: AG に「構造的近接性カーネル」を追加

```python
# 現在の AG
ag_score = cosine_sim(node_i, node_j)  # what-based のみ

# 提案: where-based を追加
structural_proximity = exp(-graph_distance(i, j)² / 2σ²)  # splatting 的
ag_score = α * cosine_sim(i, j) + (1-α) * structural_proximity
```

これにより AG が「意味的に近い」だけでなく「構造的に近い」ノードも拾える。
α は学習可能にして、where/what の最適バランスを自動決定。

### 4.2 FFN 蒸留 = Attention → Splatting 変換

Phase 2 の consolidation を「attention pattern → local kernel」への変換と解釈:

```
Phase 1: attention(q_i, k_j) で発見されたパス
Phase 2: そのパスを FFN(x_i) ≈ Σ_local kernel(x_j) · v_j に蒸留

= 「what-based な発見を where-based な直感に変換する」
```

**実験**: consolidation 後の FFN の重みが、
元の attention pattern のガウシアン近似になっているか検証。

### 4.3 GPS (Geometric Primitive Splatting for 3DGS)

既存メモ `geometric_primitives_for_3dgs_20260304.md` との接続:

```
3DGS の 59 params/Gaussian → GPS の 25-28 params
  = splatting の where 側に幾何的構造知識を入れると軽くなる
  = TSD-OCR で 11M→19K になったのと同じ原理

GPS + attention (opacity/importance を attention で学習):
  = TSD-OCR の attention-weighted splatting の 3D 版
```

---

## 5. 「解ける部分を学習させない」原則の統一

```
TSD-OCR:
  where (splatting): κ,θ を解析的に算出 → 0 params
  what (attention):  点間関係を学習 → 17K params
  合計: 19-41K で 74.38%

geDIG-Transformer:
  where (AG): β₀,β₁, 局所構造を解析的に算出 → 0 params
  what (DG):  遠方接続を学習 → 学習 params
  合計: where を解くほど what の負担が減る

3DGS → GPS:
  where: 微分幾何で Σ を決定 → 0 params (曲率から導出)
  what: albedo, roughness を学習 → 学習 params
  合計: 59→25-28 params/Gaussian (53-58% 削減)
```

**全て同じ原理: where を解析的に解き、what だけ学習させる。**

---

## 6. 今後の方向

| 項目 | 内容 | 優先度 |
|------|------|--------|
| AG の splatting 的拡張 | graph_distance カーネルの追加 | 高（実装容易） |
| Sleep consolidation の解析 | FFN 重みが attention のガウシアン近似か検証 | 中 |
| GPS 実験 | 3DGS の幾何的制約による圧縮 | 中 |
| Curl 検出の実装 | attention field の curl → 「洞察の芯」検出 | 高（geDIG の核心） |
| α の自動調整 | where/what バランスの学習 | 低（先に手動で検証） |

---

## 関連ファイル

### InsightSpike-AI
- `docs/research/dynamic_transformer_spec.md` — Dynamic Transformer 仕様
- `docs/design/gedig_transformer_architecture.md` — F最小化 Attention
- `docs/research/thinking/gedig_prediction_curl.md` — Curl 検出理論
- `docs/research/thinking/gedig_cognitive_foundation.md` — 脳との対応
- `docs/patent/geometric_primitives_for_3dgs_20260304.md` — GPS 提案

### TSD-OCR (元の発見)
- `docs/notes/theory/splatting_attention_duality_20260306.md` — 双対性の元メモ
- `experiments/p3_retinal_primitives/plan_cp_fusion.md` — C+P Fusion 設計
- `experiments/p3_retinal_primitives/models_p3e.py` — Attention 実装
- `experiments/c21_gaussian_rendering/gaussian_features.py` — Splatting 実装
