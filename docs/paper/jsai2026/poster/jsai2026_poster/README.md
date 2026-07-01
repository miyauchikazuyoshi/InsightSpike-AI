# JSAI2026 ポスター: 動的知識グラフの探索・統合を制御する統一ゲージの提案

宮内 和義(独立研究者) / JSAI2026 Session 2Yin-B-50 / 2026.06.09

## ファイル構成

```
jsai2026_poster/
├── README.md                      この説明書
├── poster.html                    単一ファイル版 (base64埋め込み、入稿・配布用)
├── poster_linked.html             外部参照版 (SVG個別編集・差分管理用)
├── figures/
│   ├── fig3_maze.png              論文図3 (迷路、PDFから抽出)
│   ├── fig4_bert.png              論文図4 (BERT箱ひげ、PDFから抽出)
│   └── svg/
│       ├── 01_motivation_brain.svg         動機の脳グラフ
│       ├── 02_formula_fepmdl.svg           統一評価関数 F + FEP/MDL 対応
│       ├── 03_figure1_dsp_branching.svg    Figure 1: ΔSP分岐
│       ├── 04_agdg_flow.svg                AG/DG 段階的判定フロー
│       ├── 06_table2_maze.svg              表2: 迷路結果
│       └── 07_table3_regularization.svg    表3: F 正則化 + 逆U字
└── src/
    ├── extract_figures_from_pdf.py  PDF→PNG抽出
    ├── extract_figures.py           poster.html → SVG分割
    └── build_poster.py              SVG分割 → poster_linked.html 再構築
```

番号は本文セクション番号と対応 (05 はHTMLテーブルのためSVG未生成、09 は展望で図なし)。

## 2つのHTML版の使い分け

### poster.html (単一ファイル版)
- 全てのSVG・画像を base64 埋め込み
- そのまま開けば完結、印刷業者への入稿・メール配布に適する
- ファイルサイズ: 約115KB

### poster_linked.html (外部参照版)
- SVG は `<object>` タグで外部参照
- PNG は `<img src="figures/...">` で外部参照
- SVG 1本ずつを編集・差分管理できる
- Gitでのレビュー・PRに適する
- 注意: ブラウザで開くときは `figures/` ディレクトリを同階層に置くこと

## 印刷

CSS に `@page { size: A0 portrait; margin: 0; }` を設定済み。
ブラウザの印刷機能でA0縦に出力可能。

- 用紙サイズ: A0 縦 (841mm × 1189mm)
- 余白: 28mm (上下) × 35mm (左右)
- 推奨: Chrome/Edge で "PDFとして保存" → 印刷業者入稿

## 図の再生成

### 論文PDFから図3・図4を抽出
```bash
python3 src/extract_figures_from_pdf.py path/to/C000993.pdf
```
トリミング座標は `src/extract_figures_from_pdf.py` 内の `CROP_BOXES` で調整可能。

### poster.html から個別SVGを抽出
```bash
python3 src/extract_figures.py
```

### 個別SVGから poster_linked.html を再構築
```bash
python3 src/build_poster.py
```

## 編集ワークフロー

1. `figures/svg/XX.svg` を直接編集 (VSCode, Inkscape など)
2. `poster_linked.html` をブラウザで開いて確認
3. 完成したら `poster.html` (base64版) を再生成

poster.html の再生成には現状スクリプト未実装。必要に応じて追加予定。

## 色の統一規則

式・図・表すべてで共通:
- ΔEPC (構造変更コスト): コーラル系 `#993C1D` / `#D85A30`
- ΔH (情報エントロピー利得): ティール系 `#0F6E56` / `#1D9E75`
- ΔSP (経路短縮利得): グリーン系 `#3B6D11`
- DG (判定・近道): パープル系 `#534AB7` / `#26215C`
- AG (注意・違和感): ΔEPC と同じコーラル系

## 参考文献

- 原論文: 宮内和義「動的知識グラフの探索・統合を制御する統一ゲージの提案」JSAI2026
- geDIG 関連コード: https://github.com/miyauchikazuyoshi/InsightSpike-AI
