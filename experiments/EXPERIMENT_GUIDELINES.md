# InsightSpike実験ガイドライン

**注意**: このガイドラインは experiments/README.md の既存ルールを補完するものです。
ディレクトリ構造とデータ管理については experiments/README.md を参照してください。

**Claudeユーザー向け**: 実験実行時は必ず `/CLAUDE.md` の実験実行ガイドラインに従ってください。
特にデータ管理ポリシーとチート禁止ルールを厳守すること。

**重要**: 設定システムが新しいPydanticベースのシステムに移行しました。実験では新しい設定システムを使用してください。

## 1. 実験設計の標準プロセス

### 1.1 実験計画書の作成
実験開始前に必ず以下を含む計画書を作成：
```markdown
# 実験名: [実験名]
## 目的
- 何を検証するか

## 仮説
- 期待される結果

## 方法
- データセット
- 評価指標
- ベースライン
- 実験条件

## 成功基準
- 定量的な基準を明記
```

### 1.2 データ準備の標準
```
experiments/
├── [experiment_name]/
│   ├── data/
│   │   ├── input/          # 元データ（変更不可）
│   │   └── knowledge_base/ # 処理済みデータ
│   ├── src/
│   │   ├── prepare_data.py # データ準備スクリプト
│   │   └── experiment.py   # 実験本体
│   └── results/
│       └── [timestamp]/    # 実行結果
```

## 2. 実験実行の標準ルール

### 2.1 データの完全性
- **チート禁止**: 答えを直接含むデータを使用しない
- **透明性**: 使用するデータの内容を明記
- **再現性**: 乱数シードを固定

### 2.2 評価の一貫性
```python
# 標準評価テンプレート
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional

from insightspike.config import load_config, InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.config.converter import ConfigConverter
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.factory import create_datastore

class StandardExperiment:
    def __init__(self, preset: str = "experiment", config_path: Optional[str] = None):
        """標準実験の初期化
        
        Args:
            preset: 使用する設定プリセット ("experiment", "development", etc.)
            config_path: カスタム設定ファイルのパス (オプション)
        """
        # 新しいconfigシステムを使用
        if config_path:
            self.config = load_config(config_path=config_path)
        else:
            self.config = load_config(preset=preset)
        
        self.datastore = create_datastore()
        self.agent = self._create_agent()
        self.validate_data()
        self._setup_logging()
    
    def _create_agent(self) -> MainAgent:
        """エージェントの作成"""
        # Pydantic設定を辞書に変換し、レガシー形式に変換
        config_dict = self.config.dict()
        legacy_config = ConfigConverter.preset_dict_to_legacy_config(config_dict)
        
        agent = MainAgent(config=legacy_config, datastore=self.datastore)
        if not agent.initialize():
            raise Exception("Failed to initialize agent")
        return agent
    
    def _setup_logging(self):
        """ロギングの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'experiment_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
    
    def validate_data(self):
        """データの妥当性チェック"""
        # 答えの直接的な含有チェック
        # データ形式の検証
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """実験メタデータの取得"""
        return {
            'timestamp': datetime.now().isoformat(),
            'config_preset': self.config.dict() if hasattr(self.config, 'dict') else str(self.config),
            'agent_type': 'MainAgent',
            'datastore_type': type(self.datastore).__name__
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """標準実験フロー"""
        results = {
            'metadata': self.get_metadata(),
            'config': self.config.dict(),  # Pydantic modelをdictに変換
            'results': [],
            'metrics': {}
        }
        # 実験実行
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """標準結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果を保存
        with open(output_dir / f'results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 設定を保存
        config_dir = Path("results/configs")
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / f'config_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(self.config.dict(), f, ensure_ascii=False, indent=2)
```

### 2.3 比較実験の標準
```yaml
# experiment_config.yaml
experiment:
  name: "RAT Comparison"
  baseline:
    - name: "Keyword RAG"
      method: "keyword_matching"
    - name: "Semantic RAG"  
      method: "embedding_similarity"
  proposed:
    name: "InsightSpike"
    method: "graph_reasoning"
  
  metrics:
    - accuracy
    - response_quality
    - spike_detection_rate
    - computational_cost
```

## 3. 結果報告の標準フォーマット

### 3.1 必須報告項目
```markdown
# 実験結果: [実験名]

## 1. 実験概要
- 実施日時
- 使用モデル
- データセット

## 2. 定量的結果
| 手法 | 精度 | 速度 | その他指標 |
|------|------|------|------------|
| ベースライン | X% | Xms | - |
| 提案手法 | X% | Xms | - |

## 3. 定性的分析
- 成功例
- 失敗例
- 原因分析

## 4. 結論
- 仮説の検証結果
- 今後の課題
```

### 3.2 ログ管理
```python
# 標準ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/experiment_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
```

## 4. 実験の品質保証

### 4.1 チェックリスト
- [ ] データに答えが直接含まれていないか
- [ ] ベースラインが適切に設定されているか
- [ ] 評価指標が明確に定義されているか
- [ ] 結果が再現可能か（シード固定等）
- [ ] エラーハンドリングが適切か

### 4.2 レビュープロセス
1. 実験計画のレビュー
2. データ妥当性の確認
3. 結果の検証
4. 改善点の記録

## 5. 共通ユーティリティ

### 5.1 実験テンプレート
```bash
# 新規実験の作成
python experiments/templates/create_experiment.py --name "new_experiment" --type "standard"

# RAT実験の作成
python experiments/templates/create_experiment.py --name "rat_test" --type "rat"

# 利用可能なタイプ: standard, rat, qa, performance
```

### 5.2 データ検証ツール
```python
# データチェッカー
python scripts/validate_data.py --data_path "data/input/" --check_answers
```

### 5.3 結果集計ツール
```python
# 複数実験の結果を集計
python scripts/aggregate_results.py --experiments "exp1,exp2,exp3"
```

## 6. 実験タイプ別ガイドライン

### 6.1 RAT実験
- 必ず5問以上でテスト
- 答えを知識ベースに含めない
- 連想の根拠を記録

### 6.2 QA実験
- 多様なドメインから質問を選択
- 回答の正確性と完全性を評価
- ハルシネーションをチェック

### 6.3 パフォーマンス実験
- ウォームアップ実行を含める
- 複数回実行して平均を取る
- システムリソースを記録