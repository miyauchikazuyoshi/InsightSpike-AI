# 組織・整理スクリプト

## 概要
プロジェクトの構造整理・実験管理・クリーンアップを行うスクリプト集

## 含まれるスクリプト

### 📁 `organize_experiments.py`
- **機能**: experiments フォルダの自動整理
- **用途**: 実験ファイルをカテゴリ別に分類・README自動生成
- **実行**: `python organize_experiments.py`

### 🔍 `analyze_experiment_importance.py`
- **機能**: 実験の重要度分析・削除候補特定
- **用途**: 重要実験の保護・不要ファイルの識別
- **実行**: `python analyze_experiment_importance.py`

### 🧹 `safe_cleanup_experiments.py`
- **機能**: 安全な実験クリーンアップ
- **用途**: 削除候補ファイルの安全削除（バックアップ付き）
- **実行**: `python safe_cleanup_experiments.py`

## 使用方法

```bash
# 実験フォルダ整理
cd /path/to/InsightSpike-AI
python scripts/organization/organize_experiments.py

# 重要度分析
python scripts/organization/analyze_experiment_importance.py

# 安全クリーンアップ
python scripts/organization/safe_cleanup_experiments.py
```

## 整理実績

- **2025年06月18日**: 50+実験ファイルを10カテゴリに整理
- **削除候補**: 2ファイル（古い設計）を安全削除
- **保護実験**: 81.6%洞察検出率の革命的実験を完全保護

## 出力場所

- レポート: `experiments/00_data_backups/`
- バックアップ: `experiments/00_data_backups/deprecated_cleanup_backup/`

---
*InsightSpike-AI Project - Organization Scripts*
