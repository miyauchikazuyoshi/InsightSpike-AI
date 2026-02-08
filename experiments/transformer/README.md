# Transformer geDIG 再現実験

**作成日**: 2026-02-03
**目的**: Phase 3 vs Phase 5 の矛盾を解明し、理論を整合させる

---

## 背景

### 既存実験の問題点

アーカイブ実験 (`_archive_before_20260201_refactor/transformer_gedig/`) で以下の矛盾が発生:

| Phase | 介入方法 | F変化 | 精度変化 | 結果 |
|-------|---------|-------|---------|------|
| **Phase 3** | Attention強制編集 | F低下 | 97% → 47% | 壊滅 |
| **Phase 5** | F正則化学習 | F低下 | 86.0% → 86.3% | 微増 |

**矛盾**: 同じ「F低下」で真逆の結果。なぜか？

### 仮説

1. **介入強度の違い**: Phase 3は破壊的、Phase 5は微調整
2. **閾値依存性**: 閾値τの選択がFの符号を変える可能性
3. **グラフ構造の保持**: Phase 5は学習で自然に構造を保持、Phase 3は強制で破壊

---

## 実験計画

### 再現実験 Phase 1: 基礎検証

**目的**: extract_and_score.py を現在のコードベースで動作確認

```bash
# smoke test
python experiments/transformer/extract_and_score.py --smoke
```

**チェック項目**:
- [ ] BERT/GPT-2 でattention抽出できる
- [ ] geDIG F計算が正常
- [ ] baseline比較でF_real < F_random

### 再現実験 Phase 2: 閾値感度分析

**目的**: 閾値τの影響を詳細に調査

| 閾値モード | τ値 | 期待 |
|-----------|-----|------|
| Percentile | 0.8, 0.85, 0.9, 0.95 | 安定 |
| Absolute | 0.01, 0.05, 0.1 | 不安定？ |

### 再現実験 Phase 3: 介入強度スイープ

**目的**: Phase 3の「強制編集」を細かく分解

| 介入強度 | 方法 | 期待 |
|---------|------|------|
| 0% | なし（ベースライン） | 97%精度 |
| 10% | 上位10%エッジをゼロ化 | ? |
| 30% | 上位30%エッジをゼロ化 | ? |
| 50% | 上位50%エッジをゼロ化 | 劇的低下？ |
| 100% | 全エッジをゼロ化 | 47%？ |

**仮説**: 精度低下は介入強度に比例し、閾値がある

### 再現実験 Phase 4: Negative Control

**目的**: F低下が本当に効いているか確認

| 正則化 | 期待結果 |
|--------|---------|
| α·F_mean（本物） | 精度微増 |
| α·random_value | 精度変化なし |

---

## ディレクトリ構造

```
experiments/transformer/
├── README.md              # この文書
├── extract_and_score.py   # Phase 1: 基礎検証
├── threshold_sweep.py     # Phase 2: 閾値感度分析
├── intervention_sweep.py  # Phase 3: 介入強度スイープ
├── train_f_regularized.py # Phase 5: F正則化（再現）
├── scripts/               # 補助スクリプト
│   └── plot_results.py
├── baselines/             # ベースライン実装
│   └── random_attention.py
└── results/               # 実験結果
    ├── phase1/
    ├── phase2/
    └── phase3/
```

---

## アーカイブからの移行

必要なファイル:

| アーカイブファイル | 移行先 | 変更点 |
|------------------|--------|--------|
| extract_and_score.py | そのまま | 依存関係の確認 |
| intervene_eval.py | intervention_sweep.py | 強度パラメータ追加 |
| train_f_regularized.py | そのまま | - |
| lambda_scan.py | threshold_sweep.py | 閾値軸に変更 |

---

## 成功基準

1. **Phase 3 vs 5 の矛盾が説明できる**
   - 介入強度閾値の特定
   - または理論の修正

2. **閾値感度が把握できている**
   - τの安全範囲が明確

3. **Negative Controlが通る**
   - F正則化は本物の効果

---

## 参考資料

- アーカイブ: `experiments/_archive_before_20260201_refactor/transformer_gedig/`
- 計画書: `docs/design/repository_review_and_plan.md` (Section 3.2)
- 理論: `docs/research/gpt_bert_gedig_perspective.md`
- 推論v2設計/実装: `experiments/transformer/inference_gedig_v2/`
