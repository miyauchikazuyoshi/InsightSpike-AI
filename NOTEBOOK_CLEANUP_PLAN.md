# InsightSpike-AI ノートブック整理計画

## 📋 現状分析

### 🔴 削除推奨（不要・重複・古い）
1. **`/notebooks/Colab_Dependency_Investigation.ipynb`** 
   - 理由: experiments/notebooks/と重複、Poetry中心のシンプル版
   - 状態: 古いアプローチ

2. **`/notebooks/InsightSpike_Colab_Setup_2025.ipynb`**
   - 理由: _fixedバージョンが存在するため不要
   - 状態: 修正前の古いバージョン

3. **`/notebooks/InsightSpike_Colab_Experiments_2025.ipynb`**  
   - 理由: _fixedバージョンが存在するため不要
   - 状態: 修正前の古いバージョン

## 🟢 保持推奨（必要・最新・役割明確）

### experiments/notebooks/ （実験・デモ用）
1. **`InsightSpike_Colab_Demo.ipynb`** ⭐ **MAIN DEMO**
   - 理由: 2025年最新対応、今回修正済み、包括的デモ
   - 役割: メインデモンストレーション
   - 特徴: T4 GPU対応、モジュールインポート修正、Poetry代替実行

2. **`InsightSpike_Bypass_Notebook.ipynb`**
   - 理由: 特殊なバイパス方式、独自の価値
   - 役割: トラブルシューティング用

3. **`geDIG_Hanoi15_demo_repository.ipynb`**
   - 理由: 特定アルゴリズムのデモ、独自価値
   - 役割: geDIGアルゴリズム専用デモ

### notebooks/ （セットアップ・安定版）
1. **`InsightSpike_Colab_Setup_2025_fixed.ipynb`** ⭐ **MAIN SETUP**
   - 理由: 最新の修正版、安定したセットアップ
   - 役割: 公式セットアップガイド
   - 特徴: 実験機能含む

2. **`InsightSpike_Colab_Experiments_2025_fixed.ipynb`**
   - 理由: 実験専用の修正版
   - 役割: 詳細実験フレームワーク

## 🔄 整理後の構造

```
notebooks/                     # 安定版・公式版
├── InsightSpike_Colab_Setup_2025_fixed.ipynb       # 公式セットアップ
├── InsightSpike_Colab_Experiments_2025_fixed.ipynb # 詳細実験
├── QUICK_START_GUIDE.md                           # クイックガイド
└── README.md                                      # 説明

experiments/notebooks/         # 実験・開発版
├── InsightSpike_Colab_Demo.ipynb                  # メインデモ（最新）
├── InsightSpike_Bypass_Notebook.ipynb             # バイパス版
└── geDIG_Hanoi15_demo_repository.ipynb           # 特殊デモ
```

## 🎯 推奨用途

### 初回ユーザー向け
1. **`notebooks/InsightSpike_Colab_Setup_2025_fixed.ipynb`** - 安定したセットアップ
2. **`experiments/notebooks/InsightSpike_Colab_Demo.ipynb`** - 最新機能のデモ

### 開発者・実験者向け  
1. **`experiments/notebooks/InsightSpike_Colab_Demo.ipynb`** - 最新開発版
2. **`notebooks/InsightSpike_Colab_Experiments_2025_fixed.ipynb`** - 詳細実験

### トラブルシューティング
1. **`experiments/notebooks/InsightSpike_Bypass_Notebook.ipynb`** - 問題回避

## 📝 削除予定ファイル詳細

### 1. `/notebooks/Colab_Dependency_Investigation.ipynb`
- **削除理由**: Poetry中心の古いアプローチ、機能が限定的
- **代替**: experiments/notebooks/InsightSpike_Colab_Demo.ipynb

### 2. `/notebooks/InsightSpike_Colab_Setup_2025.ipynb`  
- **削除理由**: _fixedバージョンで修正済み
- **代替**: InsightSpike_Colab_Setup_2025_fixed.ipynb

### 3. `/notebooks/InsightSpike_Colab_Experiments_2025.ipynb`
- **削除理由**: _fixedバージョンで修正済み  
- **代替**: InsightSpike_Colab_Experiments_2025_fixed.ipynb

## ✅ 整理実行後のメリット

1. **混乱防止**: 重複ファイルによる選択の迷いを解消
2. **保守性向上**: 維持すべきファイルが明確
3. **役割明確化**: セットアップ vs デモ vs 実験の区別
4. **最新版フォーカス**: 2025年対応の最新機能に集中
