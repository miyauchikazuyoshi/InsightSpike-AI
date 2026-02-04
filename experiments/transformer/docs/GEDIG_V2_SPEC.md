# geDIG v2 仕様案: Transformer対応

**作成日**: 2026-02-03
**目的**: 迷路とTransformerで統一的に使えるgeDIG定義

---

## 1. 背景と課題

### 1.1 現状の問題

| 成分 | 迷路（現状） | Transformer（現状） | 問題 |
|------|-------------|--------------------|----|
| **EPC** | バイナリエッジ数 | 重み差の総和 | 重み未対応/意味不明確 |
| **H** | 特徴量エントロピー | Attentionエントロピー | ドメイン解釈の相違 |
| **SP** | ホップ数ベース | 全ペア総到達性 | 希釈/ターゲット不在 |

### 1.2 根本的な違い

```
迷路:       疎グラフ → エッジ追加 → ショートカット発見
Transformer: 完全グラフ → プルーニング → ショートカット残存
```

**共通点**: 最終的に「効率的なショートカット」が存在する状態を目指す

---

## 2. 統一geDIG v2 設計

### 2.1 基本公式

```
F = ΔEPC - λ(ΔH + γΔSP)

F < 0: 良い変化（コスト低、利得高）
```

**geDIG基本原則: 延伸（ΔH>0）が利得**

- ΔH > 0: エントロピー増加 = 探索/延伸 = **利得** → F低下
- ΔSP > 0: ショートカット形成 = **利得** → F低下
- ΔEPC: 構造変化 = **コスト** → F上昇

**アブレーション用 entropy_sign パラメータ**:

| モード | s | ΔH解釈 | 用途 |
|--------|---|--------|------|
| **基本（デフォルト）** | +1 | ΔH>0（延伸）が利得 | geDIG原則 |
| アブレーション | -1 | ΔH<0（集中）が利得 | 比較実験用 |

```python
# 基本モード: entropy_sign=1 (デフォルト)
F = ΔEPC - λ(ΔH + γΔSP)   # ΔH>0 で F↓（延伸利得）

# アブレーション: entropy_sign=-1
F = ΔEPC - λ(-ΔH + γΔSP)  # ΔH<0 で F↓（集中利得）
```

### 2.2 成分の再定義

#### EPC (Edge/Path Change): 構造変化コスト

**定義**: QK構造（接続関係）の変化量

```python
# Transformer: attention weightの変化
# 閾値以上の変化のみカウント（ノイズ除去）

def compute_epc(attn_before, attn_after, threshold=0.05):
    """
    構造的に意味のあるエッジ変化を測定

    - 小さな変動はノイズとして無視
    - 大きな変化（追加/削除/強化/弱化）をカウント
    """
    diff = torch.abs(attn_after - attn_before)
    significant_changes = (diff > threshold).float()

    # 変化したエッジの割合
    epc = significant_changes.sum() / significant_changes.numel()
    return epc
```

**迷路との対応**:
- 迷路: エッジ追加/削除数
- Transformer: 有意なattention変化数

#### H (Entropy): 分布のエントロピー

**定義**: Attention分布の集中度（変更なし、解釈を明確化）

```python
def compute_entropy(attention):
    """
    Attention分布のエントロピー

    高H: 分散した注目（探索的）
    低H: 集中した注目（確信的）
    """
    # 各クエリトークンの attention 分布
    attn_norm = attention / (attention.sum(dim=-1, keepdim=True) + 1e-9)
    entropy = -(attn_norm * torch.log(attn_norm + 1e-9)).sum(dim=-1)
    return entropy.mean()
```

**解釈の統一**:
- ΔH > 0（エントロピー増加）→ 探索拡大 → 利得項で加算 → F減少
- ΔH < 0（エントロピー減少）→ 集中化 → F増加

**注**: 分類タスクでは集中化が良い場合あり → SPで補完

#### SP (Shortcut Purity): ショートカット純度 【NEW】

**定義**: アンカートークン（CLS/SEP）への経路集中度

```python
def compute_shortcut_purity(attention, anchor_indices=[0], k_ratio=0.2):
    """
    ショートカット純度: アンカーへの経路がどれだけ集中しているか

    高い値: 少数の強い経路（効率的ショートカット）
    低い値: 多数の弱い経路（ノイズ分散）

    Args:
        attention: (B, H, S, S) attention weights
        anchor_indices: ターゲットトークンの位置 [0]=CLS, [-1]=SEP
        k_ratio: 上位何%を「ショートカット」とみなすか

    Returns:
        purity: 0~1 の値（高いほど集中）
    """
    B, H, S, _ = attention.shape

    purities = []
    for anchor_idx in anchor_indices:
        # 内容トークン → アンカー への attention
        if anchor_idx == 0:
            to_anchor = attention[:, :, 1:, anchor_idx]  # CLS以外→CLS
        else:
            to_anchor = attention[:, :, :-1, anchor_idx]  # SEP以外→SEP

        # ソートして上位k%の寄与率を計算
        k = max(1, int(to_anchor.shape[-1] * k_ratio))
        top_k_values, _ = torch.topk(to_anchor, k, dim=-1)

        top_k_sum = top_k_values.sum(dim=-1)
        total_sum = to_anchor.sum(dim=-1) + 1e-9

        purity = top_k_sum / total_sum
        purities.append(purity.mean())

    return torch.stack(purities).mean()
```

**迷路との対応**:

| 状態 | 迷路 | Transformer |
|------|------|-------------|
| 初期 | ゴールへの経路なし (purity=0) | 全経路均一 (purity≈k_ratio) |
| 学習後 | ショートカット発見 (purity↑) | ショートカット残存 (purity↑) |

**ΔSP の意味**:
```
ΔSP = purity_after - purity_before

ΔSP > 0: ショートカットが形成/強化された（良い）
ΔSP < 0: 経路が分散した（悪い）
```

---

## 3. 完全なgeDIG v2 実装

```python
class GeDIGv2(nn.Module):
    """
    geDIG v2: Transformer対応版

    F = ΔEPC - λ(ΔH + γΔSP)

    - ΔEPC: 構造変化コスト（有意な変化のみ）
    - ΔH: エントロピー変化
    - ΔSP: ショートカット純度変化
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        gamma: float = 0.5,
        epc_threshold: float = 0.05,
        sp_k_ratio: float = 0.2,
        anchor_indices: list = [0],  # [0] = CLS, [-1] = SEP
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.epc_threshold = epc_threshold
        self.sp_k_ratio = sp_k_ratio
        self.anchor_indices = anchor_indices

    def forward(
        self,
        attn_before: torch.Tensor,
        attn_after: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            attn_before: (B, H, S, S) 参照状態のattention
            attn_after: (B, H, S, S) 現在のattention
            mask: (B, S) padding mask

        Returns:
            Dict with F, delta_epc, delta_h, delta_sp
        """
        # 1. ΔEPC: 有意な構造変化
        delta_epc = self._compute_epc(attn_before, attn_after, mask)

        # 2. ΔH: エントロピー変化
        h_before = self._compute_entropy(attn_before, mask)
        h_after = self._compute_entropy(attn_after, mask)
        delta_h = h_after - h_before

        # 3. ΔSP: ショートカット純度変化
        sp_before = self._compute_shortcut_purity(attn_before, mask)
        sp_after = self._compute_shortcut_purity(attn_after, mask)
        delta_sp = sp_after - sp_before

        # F値計算
        # F = ΔEPC - λ(ΔH + γΔSP)
        # 低いF = 良い変化
        F = delta_epc - self.lambda_param * (delta_h + self.gamma * delta_sp)

        return {
            "F": F,
            "F_mean": F.mean() if F.dim() > 0 else F,
            "delta_epc": delta_epc,
            "delta_h": delta_h,
            "delta_sp": delta_sp,
            "h_before": h_before,
            "h_after": h_after,
            "sp_before": sp_before,
            "sp_after": sp_after,
        }

    def _compute_epc(self, attn_before, attn_after, mask):
        """有意な構造変化のみを測定"""
        diff = torch.abs(attn_after - attn_before)

        if mask is not None:
            # padding部分を除外
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
            diff = diff * mask_2d.float()
            valid_count = mask_2d.sum() + 1e-9
        else:
            valid_count = diff.numel()

        # 閾値以上の変化をカウント
        significant = (diff > self.epc_threshold).float()
        epc = significant.sum() / valid_count

        return epc

    def _compute_entropy(self, attention, mask):
        """Attention分布のエントロピー"""
        if mask is not None:
            # padding部分をマスク
            mask_2d = mask.unsqueeze(1).unsqueeze(2).float()
            attention = attention * mask_2d

        attn_norm = attention / (attention.sum(dim=-1, keepdim=True) + 1e-9)
        entropy = -(attn_norm * torch.log(attn_norm + 1e-9)).sum(dim=-1)

        if mask is not None:
            # 有効なトークンのみで平均
            valid_mask = mask.unsqueeze(1).float()
            entropy = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-9)
        else:
            entropy = entropy.mean()

        return entropy

    def _compute_shortcut_purity(self, attention, mask):
        """アンカーへの経路集中度"""
        B, H, S, _ = attention.shape

        purities = []
        for anchor_idx in self.anchor_indices:
            # anchor_idx が負の場合の処理
            if anchor_idx < 0:
                anchor_idx = S + anchor_idx

            # 内容トークン → アンカー への attention
            # アンカー自身は除外
            indices = [i for i in range(S) if i != anchor_idx]
            to_anchor = attention[:, :, indices, anchor_idx]  # (B, H, S-1)

            if mask is not None:
                # アンカー以外のmask
                content_mask = torch.cat([
                    mask[:, :anchor_idx],
                    mask[:, anchor_idx+1:]
                ], dim=1) if anchor_idx < S-1 else mask[:, :anchor_idx]
                to_anchor = to_anchor * content_mask.unsqueeze(1).float()

            # 上位k%の寄与率
            k = max(1, int(to_anchor.shape[-1] * self.sp_k_ratio))
            top_k_values, _ = torch.topk(to_anchor, k, dim=-1)

            top_k_sum = top_k_values.sum(dim=-1)
            total_sum = to_anchor.sum(dim=-1) + 1e-9

            purity = (top_k_sum / total_sum).mean()
            purities.append(purity)

        return torch.stack(purities).mean() if purities else torch.tensor(0.0)
```

---

## 4. 期待される動作

### 4.1 学習進行時の各成分

| エポック | ΔEPC | ΔH | ΔSP | F | 解釈 |
|---------|------|-----|-----|---|------|
| 初期 | 0 | 0 | 0 | 0 | 基準 |
| 序盤 | 高 | 負(集中化) | 正(純化) | **負(良い)** | 構造形成中 |
| 中盤 | 中 | 負 | 正 | **負(良い)** | 安定した改善 |
| 終盤 | 低 | 小 | 小 | ≈0 | 収束 |
| 過学習 | 高 | 極端 | 負? | **正(悪い)** | 崩壊 |

### 4.2 介入実験の期待

```
Positive介入（F↓方向）:
- ΔSP↑（ショートカット強化）
- 精度向上を期待

Negative介入（F↑方向）:
- ΔSP↓（ショートカット弱化）
- 精度低下を期待
```

---

## 5. 初期状態の理論的考察

### 5.1 ドメインによる初期状態の違い

| ドメイン | 初期状態 | エントロピー | 学習方向 |
|----------|----------|--------------|----------|
| **迷路** | 空グラフ（疎） | H ≈ 0 | 構築 → H↑ |
| **Transformer（ランダム初期化）** | 一様分布（発散） | H = max | プルーニング → H↓ |
| **Transformer（事前学習済み）** | 既に構造化 | H = mid | 微調整 → H最適化 |

### 5.2 理論的予測

**一様分布のエントロピー**:
```
H_max = log(S)  where S = sequence length
例: S=128 → H_max ≈ 4.85
```

**観測された値（事前学習済みDistilBERT）**:
```
H_initial ≈ 1.76  (H_max の約36%)
SP_initial ≈ 0.87 (既にCLSへの経路集中あり)
```

**解釈**: 事前学習済みモデルは「発散状態」ではなく「汎用的に構造化された状態」から開始する。

### 5.3 geDIG解釈の修正

| フェーズ | entropy_sign | 理由 |
|----------|--------------|------|
| 事前学習（ゼロから） | +1 | 発散→構造化、延伸が利得 |
| Fine-tuning | -1 | 汎用→特化、集中化が利得 |

両フェーズでgeDIGが機能すれば、より一般的な理論となる。

---

## 6. 実験計画（改訂版）

### Phase 0: Pythiaチェックポイントによる学習過程全体の観察 【軽量・数分】
**目的**: 学習初期〜完了までのgeDIG成分の推移を観察

**使用モデル**: [Pythia (EleutherAI)](https://huggingface.co/EleutherAI/pythia-70m)
- 70M〜12Bパラメータ
- 154個のチェックポイント（step0〜step143000）
- 学習ダイナミクス研究用に設計

**実験内容**:
- [ ] step0（初期化直後）で H ≈ H_max を確認
- [ ] 主要チェックポイント（step0, 1000, 10000, 50000, 143000）で H, SP を計算
- [ ] 層ごとの H, SP 推移を可視化
- [ ] 期待: H↓（集中化）、SP↑（ショートカット形成）

```python
from transformers import AutoModelForCausalLM

# チェックポイント一覧
checkpoints = ["step0", "step1000", "step10000", "step50000", "step143000"]

for ckpt in checkpoints:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision=ckpt,
        output_attentions=True
    )
    # H, SP を計算
    ...
```

**利点**:
- 自前で学習不要（リソース節約）
- 学習過程全体を網羅的に観察可能
- 「発散→集中」仮説の純粋な検証

### Phase 1-A: Pythia層ごと分析 【軽量・数分】
**目的**: 層ごとの役割分化を観察

- [ ] 各チェックポイントで層ごとのH, SP, Fを計算
- [ ] 低層 vs 高層で異なるダイナミクスがあるか確認
- [ ] 「目標構造」が層によって異なるか検証

**期待される発見**:
| 層 | 役割仮説 | 予測されるパターン |
|----|----------|-------------------|
| 低層 | 構文的パターン | 早期に収束、局所的SP↑ |
| 中層 | 意味的統合 | 中間的な変化 |
| 高層 | タスク関連 | 後期に変化、CLS/EOSへのSP↑ |

### Phase 1-B: 事前学習済みでのFine-tuning 【軽量・完了済み】
**目的**: 「汎用→特化」プロセスの実用検証

- [x] microscopic観察実験
- [x] 介入実験（Positive/Negative/Baseline）
- [x] entropy_sign=-1 で Positive > Negative を確認

**結果**: entropy_sign=-1 で geDIG F が有効な学習シグナルとなることを確認

**重要な発見**: 事前学習済みモデルは既に構造化されている（H ≈ 1.76、理論最大の36%）

### Phase 2: 統合分析・目標構造の定式化
**目的**: 学習フェーズごとの「目標」を明確化

| フェーズ | 初期状態 | 目標状態 | entropy_sign |
|----------|----------|----------|--------------|
| 事前学習（Pythia観察） | H=max, SP=low | H=mid, SP=high | +1（延伸→構造化） |
| Fine-tuning | H=mid, SP=mid | H=low, SP=high（タスク特化） | -1（集中化） |

**分析内容**:
- [ ] Phase 0/1-A（Pythia）と Phase 1-B（Fine-tuning）の結果比較
- [ ] 「目標構造」= 学習完了時のH, SP値として定義
- [ ] F値が「目標への距離」として機能するか検証
- [ ] 最適な entropy_sign 選択基準の定式化

### Phase 3: 迷路との統一検証
**目的**: ドメイン横断的な理論の確立

- [ ] 迷路実験でSP定義を更新（アンカーベース: ゴールノード）
- [ ] 両ドメインで同一公式が機能するか確認
- [ ] 統一理論の文書化

**期待される結論**:
```
geDIG F = 構造最適化の汎用指標
- 迷路: 空グラフ → ゴールへのショートカット形成
- Transformer事前学習: 一様分布 → 効率的な情報経路形成
- Transformer Fine-tuning: 汎用構造 → タスク特化構造
```

---

## 7. 単体テスト結果

```
[Test 2] Uniform vs Shortcut (few tokens → CLS)
  BASELINE (entropy_sign=1):
    F_mean: 0.2545
    delta_epc: 0.2000 (structure changed = cost)
    delta_h: -0.1632 (concentrated = reduces benefit in baseline)
    delta_sp: 0.2173 (shortcuts formed = benefit)

[Test 3] Shortcut vs Uniform (reverse = losing shortcuts)
  F_mean: 0.1455 (> 0 = bad change)
  delta_sp: -0.2173 (shortcuts lost)

[Test 6] Baseline vs Ablation
  BASELINE (entropy_sign=1):  F = 0.2545 (ΔH<0 hurts score)
  ABLATION (entropy_sign=-1): F = -0.0718 (ΔH<0 helps score)
```

**結果の解釈**:
- **基本モード**: 延伸（ΔH>0）が利得。集中化（ΔH<0）は利得を減らす
- **アブレーション**: 集中化も利得として扱うと F < 0 になる
- ショートカット形成（ΔSP>0）は両モードで利得
- 実験で両モードを比較し、どちらがTransformer学習に適するか検証

---

## 8. 変更履歴

- 2026-02-03: 初版作成
- 2026-02-03: entropy_sign パラメータ追加（分類/探索モード切替）
- 2026-02-03: 初期状態の理論的考察を追加（セクション5）
  - ランダム初期化 vs 事前学習済みの違いを明確化
  - 実験計画をPhase 0/1-A/1-B/2/3に再構成
  - 事前学習済みモデルでの実験結果の正しい解釈を追加
- 2026-02-03: Pythiaチェックポイントを活用した実験設計に改訂
  - Phase 0: Pythiaで学習過程全体を観察（自前学習不要）
  - Phase 1-A: 層ごと分析の追加
  - Phase 2: 目標構造の定式化を追加
  - 「発散→集中」仮説の純粋な検証が可能に
