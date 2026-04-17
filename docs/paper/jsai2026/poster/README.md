# JSAI 2026 Poster

**採択論文**: [../v3/main.pdf](../v3/main.pdf) (v3)  
**採択論文のタイトル**: 動的知識グラフの探索・統合を制御する統一ゲージの提案 — *Gauge what Knowledge Graph needs.*  

## ファイル構成

```
poster/
├── README.md                       このファイル
├── poster.tex                      ポスター本体 (tikzposter, A0 縦)
└── figs/
    └── figure1_hybrid.tex          Figure 1 (orbit + matchstick 統合版)
```

## ビルド方法

### 推奨: lualatex (現行設定)

```bash
cd docs/paper/jsai2026/poster
lualatex poster.tex
lualatex poster.tex             # 2 回目は cross-reference 安定化のため (任意)
```

→ `poster.pdf` が直接生成されます。

### なぜ lualatex か — uplatex + dvipdfmx では不具合

**tikzposter + uplatex + dvipdfmx の既知の相性問題**:
- tikzposter はブロック描画に PostScript 特殊命令 (`ps::`) を使用
- dvipdfmx はこれを解釈できず「`Unknown token "restore"`」警告を出す
- 結果: **タイトル以外のブロックが全て消失**してしまう

→ **lualatex** は PDF を直接出力するためこの問題を回避できる。
   日本語は `luatexja` パッケージで対応済み (HaranoAji フォント使用)。

### uplatex 環境で試したい場合 (非推奨)

`poster.tex` の日本語設定を差し替え、さらに tikzposter のブロック描画を
avoid する調整が必要。現実的には lualatex を推奨。

### 依存パッケージ

- `tikzposter` (ポスタークラス)
- `luatexja` (lualatex 用 日本語組版)
- `amsmath`, `amssymb`, `booktabs`, `array`
- `tikz` + libraries: `arrows.meta`, `positioning`, `shapes.geometric`, `fit`, `calc`
- `hyperref`, `xcolor`, `graphicx`

MacTeX / TeX Live 2025 以上であれば標準で揃っています。

## 構成 (7 セクション, A0 縦)

```
[Title]                             Gauge what Knowledge Graph needs.
[1. Introduction]                   When 問題 + 貢献 3 点
[2. Why three terms?]               ★Figure 1 hybrid (核心)
[3. Formulation]                    F 式 + entropy_sign
[4. Experiments]                    迷路 98% + Transformer 4.6% 差
[5. Discussion]                     SP → β₁ + OCR 外部実証
[6. Limitations]                    誠実な限界明示
[7. Conclusion & Resources]         Future Work + QR/Links
```

## Figure 1 について

**統合ハイブリッド図**: orbit (空間フレーム) + matchstick (3 ケース) を 1 枚に統合。

### 構造

- **外側の赤い円** = Multi-hop Reach (DG 領域、`ΔSP` で評価)
- **内側の青い円** = 0-hop Orbit (AG 領域、`ΔEPC`, `ΔH` で評価)
- **中心の赤星** = Query $Q$ (Transformer なら CLS、迷路ならゴール)

### 3 ケース

| Case | 配置領域 | 意味 | 判定 |
|---|---|---|---|
| A: Insight | Multi-hop 圏 (外側円内) | 赤太線で新短絡 (`ΔSP=+1`) | DG 発火 → 受け入れ (F<0) |
| B: 力仕事 | 0-hop 圏 (内側円内) | 無関係な編集 (`ΔSP=0`) | 発火なし → 保留 (F≈0) |
| C: 崩壊 | Multi-hop 圏 | 既存短絡が消失 (`ΔSP=-1`) | 逆発火 → 拒否 (F>0) |

### 設計意図

- **1 枚で AG/DG 二段ゲート構造を示す**: 2 つの円で役割分担が視覚化
- **3 ケースが orbit 内の位置で意味付けされる**: 空間的な配置 = 意味論
- **クエリ $Q$ を明示**: 論文 §4 の SP 定義と完全整合
- **採択論文 (SP ベース) に忠実**: β₁ 拡張は Discussion セクションで議論

## 採択論文との関係

ポスターは **v3 採択版 (SP + entropy_sign + 介入実験)** の主張を忠実に反映:

- §1-4: 採択論文の範囲内
- §5 Discussion: SP → β₁ 一般化と OCR 実証 (採択後の自然な拡張として明示)
- §6 Limitations: 採択論文の §5.3 相当 + 誠実な範囲限定

## ポスター発表シナリオ (4 分立ち話)

1. **30 秒**: タイトル指差し「Gauge what Knowledge Graph needs — When 問題です」
2. **1 分**: Figure 1 の Query $Q$ と 2 つの円を説明 (AG/DG の必然性)
3. **1 分**: Case A/B/C で 3 判定 (受け入れ/保留/拒否)
4. **1 分**: Transformer 介入実験 4.6% 差 → 因果的シグナルを実証
5. **30 秒**: 質問対応 (β₁ への拡張、OCR 実証等)

## トラブルシューティング

### 「タイトル以外のブロックが消えている」

- **原因**: uplatex + dvipdfmx を使っている場合、tikzposter の PostScript 特殊命令が dvipdfmx で解釈できない既知の問題
- **対処**: `lualatex poster.tex` を使う (現行設定)

### 日本語が出ない

- `luatexja` パッケージがインストール済みか確認:
  `kpsewhich luatexja.sty`
- TeX Live 2020 以上であれば標準装備

### Figure 1 が表示されない

- `figs/figure1_hybrid.tex` のパス確認
- TikZ ライブラリ `arrows.meta` が読み込まれているか確認

### tikzposter のテーマ調整

色やレイアウトを変えたい場合, `poster.tex` 上部の以下を変更:

```latex
\usetheme{Simple}              % Default, Simple, Rays, Basic, Envelope, Wave, Board, Autumn, Desert, Wake
\usecolorstyle{Default}        % Default, Australia, Britain, Sweden, ...
\useblockstyle{Default}        % Default, Basic, Minimal, Envelope, Corner, Slide, TornOut
```

### Figure 1 のラベル重なり調整

`figs/figure1_hybrid.tex` 内の `\node` 座標を調整:
- Case A/B/C ボックスの座標 `at (x, y)`
- 領域ラベル (`Multi-hop Reach`, `0-hop Orbit`) の座標
- 凡例ボックスの座標

調整後は `lualatex poster.tex` で再ビルド。

## 関連ドキュメント

- **採択論文**: [../v3/main.pdf](../v3/main.pdf)
- **v3 構成**: [../v3/README.md](../v3/README.md), [../v3/PAPER_OUTLINE.md](../v3/PAPER_OUTLINE.md)
- **Part 1 コア理論統合**: [../../../research/gedig_core_theory_unified.md](../../../research/gedig_core_theory_unified.md)
- **1-page Overview**: [../../../research/overview.md](../../../research/overview.md)
- **β₁ 次元フリー性** (Discussion 背景): [../../../research/thinking/insight_beta1_dimension_free.md](../../../research/thinking/insight_beta1_dimension_free.md)
- **連続・確率パラダイム批判** (思想的背景): [../../../research/thinking/insight_continuous_probabilistic_paradigm_critique.md](../../../research/thinking/insight_continuous_probabilistic_paradigm_critique.md)

## 発表後の TODO

- [ ] 質疑応答メモを `discussion_notes.md` に記録
- [ ] 発表スライドを `slides/` サブディレクトリに追加 (必要なら)
- [ ] ビデオ記録があればリンク追加
