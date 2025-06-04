# Experiments Directory

このディレクトリには、InsightSpike-AIの実験、テスト、および作業用ファイルが含まれています。

## ディレクトリ構造

### 実験カテゴリ
- `rl_experiments/` - 強化学習・迷路探索実験
- `educational_demos/` - 教育用デモンストレーション
- `analysis_tools/` - 性能分析・評価ツール

### 作業ディレクトリ
- `notebooks/` - Jupyter Notebookによる実験・分析
- `outputs/` - 実験結果の出力ファイル（JSON、画像等）
- `results/` - 実験結果アーカイブ
- `data/` - 実験用データファイル

## 重要事項

### 含まれるファイル
- 重要なデモ用Notebook（`InsightSpike_Colab_Demo.ipynb`等）
- ベースライン実験結果
- 再現可能な実験設定

### 除外されるファイル（.gitignore）
- 一時的な実験結果（`outputs/*.json`）
- 作業用・調査用Notebook（`*_Investigation.ipynb`等）
- テスト用スクリプト（`test_*.py`）

## 使用方法

```bash
# 実験環境のセットアップ
cd experiments
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r ../requirements.txt

# Jupyter Notebookの起動
jupyter notebook
```

## 注意事項

- 実験結果は一時的なものとして扱われます
- 重要な結果は`documentation/`ディレクトリに永続化してください
- 大容量ファイルは自動的に除外されます
