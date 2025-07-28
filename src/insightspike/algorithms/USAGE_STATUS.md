# Algorithm Usage Status (2025-01-27 更新)

## ✅ 実際に使用されているアルゴリズム

### 統一実装（最新）
- **gedig_core.py** - すべてのgeDIG計算の中核実装
- **gedig_calculator.py** - gedig_coreのラッパー（後方互換性）

### アクティブに使用中
- **metrics_selector.py** - アルゴリズム選択の中核（gedig_core使用）
- **information_gain.py** - improved_gedig_metrics、graph_metrics、episodic/hybrid_episode_splitterで使用
- **graph_structure_analyzer.py** - improved_gedig_metricsで使用
- **pyg_adapter.py** - metrics_selectorで条件付き使用（PyG環境用）

### 内部依存として使用中（information_gain.py経由）
- **entropy_calculator.py** - information_gain.pyが使用
- **structural_entropy.py** - information_gain.pyが使用
- **improved_similarity_entropy.py** - information_gain.pyが使用

## 🗑️ legacyに移動すべきファイル（gedig_coreに統合済み）

- **normalized_ged.py** - gedig_coreに統合済み
- **entropy_variance_ig.py** - gedig_coreに統合済み
- **multihop_gedig.py** - gedig_coreに統合済み
- **local_information_gain_v2.py** - gedig_coreに統合済み

## 📝 現在の使用パターン

1. **gedig_core経由（推奨）**
   - すべてのgeDIG計算はgedig_coreを使用
   - MetricsSelectorもデフォルトでgedig_coreを使用
   - GeDIGCalculatorはgedig_coreのラッパー

2. **直接import（レガシー用途）**
   - information_gain.py - 複数のモジュールが依存
   - graph_structure_analyzer.py - improved_gedig_metricsが使用
   - pyg_adapter.py - PyTorch Geometric環境で必要

3. **内部依存関係**
   ```
   information_gain.py
   ├── entropy_calculator.py
   ├── structural_entropy.py
   └── improved_similarity_entropy.py
   ```

## 🔍 推奨アクション

1. normalized_ged.py、entropy_variance_ig.py、multihop_gedig.py、local_information_gain_v2.pyをlegacyに移動
2. information_gain.pyとその依存関係は維持（他モジュールが使用中）
3. pyg_adapter.pyは維持（PyG環境で必要）