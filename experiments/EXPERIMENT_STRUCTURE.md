# Experiment Management Structure

## 現在の問題点
1. 実験がルートディレクトリに散在（experiment_5, experiment_6, etc.）
2. experiments/フォルダ内の構造が不統一
3. データとコードが混在
4. 結果の保存場所がバラバラ

## 提案する標準構造

```
experiments/
├── README.md                    # 実験管理ガイド
├── EXPERIMENT_REGISTRY.md       # 実験一覧と概要
├── template/                    # 実験テンプレート
│   ├── experiment.py
│   ├── config.yaml
│   └── README.md
│
├── 2025-07-06_experiment_12_scalable_graph/    # 実験ごとのディレクトリ
│   ├── README.md               # 実験の説明
│   ├── config.yaml             # 実験設定
│   ├── code/                   # 実験コード
│   │   ├── main.py
│   │   └── utils.py
│   ├── data/                   # 実験用データ（※メインのdataとは別）
│   │   ├── input/              # 入力データ
│   │   └── output/             # 出力データ
│   ├── results/                # 実験結果
│   │   ├── metrics.json
│   │   ├── plots/
│   │   └── logs/
│   └── notebooks/              # 分析用ノートブック
│       └── analysis.ipynb
│
└── archive/                    # 古い実験のアーカイブ
    └── 2025-06-*_*/
```

## 命名規則

実験ディレクトリ名：`YYYY-MM-DD_experiment_<番号>_<簡潔な説明>`

例：
- `2025-07-06_experiment_12_scalable_graph`
- `2025-07-05_experiment_11_rag_enhancement`

## 実験実行の標準フロー

```python
# experiments/template/experiment.py

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class ExperimentBase:
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(__file__).parent
        self.results_dir = self.exp_dir / "results"
        self.data_dir = self.exp_dir / "data"
        
        # ディレクトリ作成
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
    def backup_main_data(self):
        """メインのdataディレクトリをバックアップ"""
        main_data = PROJECT_ROOT / "data"
        backup_path = self.data_dir / f"backup_{self.timestamp}"
        shutil.copytree(main_data, backup_path)
        print(f"Backed up main data to {backup_path}")
        
    def save_results(self, results: dict):
        """結果を保存"""
        results_file = self.results_dir / f"results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")
```

## データ管理ルール

1. **実験データは experiments/内で完結**
   - 入力データ: `experiments/<実験名>/data/input/`
   - 出力データ: `experiments/<実験名>/data/output/`
   - 結果: `experiments/<実験名>/results/`

2. **メインデータ（data/）への影響**
   - 読み取りのみ推奨
   - 変更が必要な場合は必ずバックアップ
   - 実験終了後は元に戻す

3. **結果の保存**
   - すべての結果は `experiments/<実験名>/results/` に保存
   - タイムスタンプ付きファイル名を使用
   - メトリクス、ログ、可視化を分離

## 移行計画

### Phase 1: 新規実験から適用
- 今後の実験はすべてこの構造に従う
- テンプレートを使用して開始

### Phase 2: 既存実験の整理
```bash
# ルートの実験を移動
mv experiment_5 experiments/2025-07-04_experiment_5_scalability/
mv experiment_6 experiments/2025-07-04_experiment_6_huggingface/
# ... etc
```

### Phase 3: データの整理
- `data/`から実験固有のファイルを各実験ディレクトリに移動
- `data/`はシステムのコアデータのみを保持

## 利点

1. **整理された構造** - 実験が探しやすい
2. **独立性** - 実験間の干渉を防ぐ
3. **再現性** - 各実験の設定とデータが保存される
4. **アーカイブ可能** - 古い実験を簡単に移動/削除
5. **共同作業** - 実験ごとに担当者を分けやすい