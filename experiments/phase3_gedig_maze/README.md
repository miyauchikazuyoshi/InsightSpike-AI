# Phase 3: GEDIG迷路実験

## 概要

粘菌インスパイアGEDIGアルゴリズムとA*等の従来手法を迷路上で厳密比較し、洞察スパイクや情報利得を可視化・定量評価する実験です。

## 目標

- **60%試行回数削減**
- **3倍高速収束**
- **95%成功率**

## 実装された機能

### ✅ 完了した実装

1. **迷路生成システム**
   - Recursive Backtrackingアルゴリズムによる壁構造迷路
   - 複数サイズ（11x11、21x21、15x15、31x31）の迷路自動生成
   - 経路アクセシビリティ保証

2. **アルゴリズム比較**
   - A*アルゴリズム
   - Dijkstraアルゴリズム  
   - 強化学習アルゴリズム
   - 遺伝的アルゴリズム
   - **SlimeMold_GEDIGアルゴリズム**（粘菌インスパイア）

3. **理論的フレームワーク**
   - エッジ分布エントロピー計算
   - コサイン類似度GED（Graph Edit Distance）
   - 情報利得（Information Gain）計算
   - 洞察スパイク検出

4. **可視化・レポート**
   - 性能比較グラフ（PNG）
   - 経路比較アニメーション（GIF）
   - 詳細性能レポート（Markdown）
   - 実験データ（CSV）

## ファイル構成

```text
phase3_gedig_maze/
├── gedig_maze_experiment.py    # メイン実験スクリプト
├── final_results/              # 最終実験結果
│   ├── gedig_performance_report.md
│   ├── gedig_maze_results.csv
│   ├── gedig_performance_visualization.png
│   └── path_comparison_RecursiveMaze_11x11.gif
├── results/                    # 全実験結果
├── data/                      # 実験データ
├── archive/                   # 開発用ファイル
└── README.md                  # このファイル
```

## 実行方法

### 基本実行

```bash
python gedig_maze_experiment.py
```

### GIF アニメーション付き実行

```bash
python gedig_maze_experiment.py --animate
```

### クイックテスト

```bash
python gedig_maze_experiment.py --quick --animate
```

### デバッグモード

```bash
python gedig_maze_experiment.py --debug --animate
```

## 実験結果

### 主要成果

- **GEDIGアルゴリズム**: 試行回数を従来手法の5分の1に削減
- **洞察スパイク**: 効率的な経路発見を実現
- **情報利得**: 学習プロセスの定量的評価が可能

### 生成される出力

1. **性能レポート** (`gedig_performance_report.md`)
   - アルゴリズム別詳細比較
   - 目標達成度評価
   - 統計的分析結果

2. **可視化** (`gedig_performance_visualization.png`)
   - 試行回数、収束時間、成功率の棒グラフ
   - アルゴリズム間の性能比較

3. **アニメーション** (`path_comparison_*.gif`)
   - A*とGEDIGの経路探索過程を並べて比較
   - 段階的な経路構築アニメーション

4. **データ** (`gedig_maze_results.csv`)
   - 全実験データの詳細記録
   - 統計分析用の構造化データ

## 技術的特徴

### 粘菌インスパイアGEDIGアルゴリズム

- **エッジ分布エントロピー**: グラフ構造の複雑さを定量化
- **コサイン類似度GED**: 効率的なグラフ編集距離計算
- **情報利得**: 探索ステップごとの有用性評価
- **洞察スパイク**: 突破的発見の検出

### 実験設計

- **データ安全性**: 自動バックアップ・ロールバック機能
- **再現性**: 固定シードによる決定論的実行
- **スケーラビリティ**: 複数迷路サイズでの性能評価

## 開発経緯

1. **Phase 1**: 基本アルゴリズム実装
2. **Phase 2**: 理論的フレームワーク構築  
3. **Phase 3**: 迷路実験・可視化完成
4. **Phase 4**: 統合評価・最適化（予定）

## 依存関係

- Python 3.8+
- numpy, pandas, matplotlib
- imageio (GIF生成用)
- その他: requirements.txt参照

---

**実験完了日**: 2025年6月22日  
**ステータス**: ✅ 完成・運用可能
