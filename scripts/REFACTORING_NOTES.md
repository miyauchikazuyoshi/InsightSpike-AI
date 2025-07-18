# Scripts Refactoring Notes

## リファクタリング影響と対応状況

### 更新済みスクリプト ✅

1. **production/system_validation.py**
   - `insightspike.config.get_config` → `insightspike.config.loader.load_config`
   - `core.agents.main_agent` → `implementations.agents.main_agent`
   - プリセットベースの設定読み込みに変更

2. **testing/safe_component_test.py**
   - 新しい設定システムに対応
   - Embedderクラスの直接利用に変更

3. **testing/test_complete_insight_system.py**
   - レガシーシステムへの依存を削除
   - 新しいMainAgentとDataStoreを使用するように完全に書き直し

4. **validation/complete_system_validation.py**
   - 新しい設定システムに対応
   - FallbackEmbedder → Embedder に変更

5. **benchmarks/performance_suite.py**
   - インポートパスを修正
   - GraphEditDistance → graph_importance モジュールから
   - InsightDetector → EurekaSpike を使用

### 更新不要なスクリプト ✅

以下のスクリプトはInsightSpikeのコアモジュールを使用していないため更新不要：

- `monitoring/production_monitor.py` - 独立したモニタリングシステム
- `setup_models.py` - モデルダウンロードのみ

### 主な変更点

1. **設定システム**
   ```python
   # 旧
   from insightspike.config import get_config
   config = get_config()
   
   # 新
   from insightspike.config.loader import load_config
   from insightspike.config.presets import ConfigPresets
   config = load_config(preset="development")
   ```

2. **MainAgent パス**
   ```python
   # 旧
   from insightspike.core.agents.main_agent import MainAgent
   
   # 新
   from insightspike.implementations.agents.main_agent import MainAgent
   ```

3. **Embedder**
   ```python
   # 旧
   from insightspike.processing.embedder import get_model_singleton
   
   # 新
   from insightspike.processing.embedder import Embedder
   embedder = Embedder(config=config)
   ```

### 推奨事項

重要度の高いスクリプトから順次更新を進めることを推奨：
1. production/ 配下のスクリプト（本番運用に影響）
2. validation/ 配下のスクリプト（品質保証に影響）
3. その他のスクリプト