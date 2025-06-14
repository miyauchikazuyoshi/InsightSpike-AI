# 📓 Experiments Notebooks

このディレクトリには、InsightSpike-AIの最新実験とデモンストレーション用ノートブックが含まれています。

## 📂 ノートブック一覧

### 🚀 メインデモ（推奨）
- **`InsightSpike_Colab_Demo.ipynb`** - **2025年最新対応のメインデモ**
  - T4 GPU対応、モジュールインポート修正済み
  - Poetry代替実行システム
  - 包括的な機能デモンストレーション

### 🔧 特殊用途
- **`InsightSpike_Bypass_Notebook.ipynb`** - **トラブルシューティング用バイパス版**
  - 通常セットアップが困難な場合の代替手段
  - 最小限の依存関係で動作

- **`geDIG_Hanoi15_demo_repository.ipynb`** - **geDIGアルゴリズム専用デモ**
  - Hanoi Tower問題での洞察検出実演
  - アルゴリズム研究用

- **`Colab_Dependency_Investigation.ipynb`** - **依存関係調査用**
  - Poetry中心のシンプルなセットアップ
  - デバッグ・調査用途

## 🎯 推奨利用順序

### 初回ユーザー
1. `/notebooks/InsightSpike_Colab_Setup_2025_fixed.ipynb` (公式セットアップ)
2. `InsightSpike_Colab_Demo.ipynb` (メインデモ)

### 開発者・研究者
1. `InsightSpike_Colab_Demo.ipynb` (最新機能)
2. `/notebooks/InsightSpike_Colab_Experiments_2025_fixed.ipynb` (詳細実験)

### トラブルシューティング
1. `InsightSpike_Bypass_Notebook.ipynb` (代替方法)
2. `Colab_Dependency_Investigation.ipynb` (調査・デバッグ)

## 💡 特徴

- **2025年Colab対応**: NumPy 2.x、PyTorch 2.x環境での動作確認済み
- **GPU最適化**: T4 GPU環境での最適な性能
- **エラー処理**: 包括的なトラブルシューティング
- **モジュラー設計**: 用途別に分離された明確な役割
