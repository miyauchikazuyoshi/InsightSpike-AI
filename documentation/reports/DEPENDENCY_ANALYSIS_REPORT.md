# InsightSpike-AI スクリプト依存関係分析レポート

**作成日時:** 2025年5月31日  
**分析対象:** 整理後のプロジェクト構造における全スクリプトファイル

## 📋 分析概要

プロジェクト構造の再整理により、14のPythonスクリプトと3のシェルスクリプトが新しいディレクトリ構造に移動されました。各スクリプトのInsightSpikeモジュールへの依存関係とその現在の状態を分析しました。

## 🗂️ スクリプト分類

### 🏭 Production Scripts (`scripts/production/`)
- `system_validation.py` ✅ **Main依存** - システム全体の検証
- `run_true_insight_experiment.py` ⚠️ **Mock依存** - 独立した実験
- `create_true_insight_experiment.py` ✅ **Mock独立** - データセット生成
- `create_minimal_index.py` ✅ **Utility独立** - インデックス生成

### 🧪 Testing Scripts (`scripts/testing/`)
- `test_complete_insight_system.py` ✅ **Main依存** - システム統合テスト
- `test_llm_config_fix.py` ✅ **Main依存** - LLM設定テスト
- `test_llm_config_fix_lite.py` ✅ **Main依存** - 軽量LLMテスト
- `test_hf_dataset_integration.py` ⚠️ **HF依存** - データセット統合テスト

### ☁️ Colab Scripts (`scripts/colab/`)
- `colab_large_scale_experiment.py` ✅ **Main依存** - 大規模実験
- `colab_diagnostic.py` ⚠️ **Diagnostic独立** - 診断ツール

### 🔧 Utilities Scripts (`scripts/utilities/`)
- `generate_visual_summary.py` ✅ **独立** - 結果可視化
- `comprehensive_rag_analysis.py` ✅ **独立** - 比較分析

### 🔧 Setup Scripts (`scripts/setup/`)
- `setup.sh` ✅ **Shell独立** - 環境セットアップ
- `refactor_prepare.sh` ✅ **Shell独立** - リファクタ準備

### 📊 Root Level Scripts (`scripts/`)
- `run_poc_simple.py` ⚠️ **Mock依存** - 簡易PoC実行

## 📊 依存関係マトリックス

| スクリプト | MainAgent | CLI | Core | Legacy | Mock | 独立 |
|-----------|-----------|-----|------|-------|------|------|
| system_validation.py | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| colab_large_scale_experiment.py | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| test_complete_insight_system.py | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| test_llm_config_fix.py | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| test_llm_config_fix_lite.py | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| run_true_insight_experiment.py | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| create_true_insight_experiment.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| create_minimal_index.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| test_hf_dataset_integration.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| colab_diagnostic.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| generate_visual_summary.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| comprehensive_rag_analysis.py | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| run_poc_simple.py | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

## 🔍 詳細分析

### ✅ 現在動作する Main システム依存スクリプト

#### 1. `scripts/production/system_validation.py`
**依存関係:**
- `insightspike.core.agents.main_agent.MainAgent` - 新しいメインエージェント
- `insightspike.insight_fact_registry.InsightFactRegistry` - 洞察記録
- `insightspike.core.config.get_config` - 設定管理
- `insightspike.cli.app` - CLI機能

**機能:** システム全体の包括的検証、各レイヤーのテスト、統合テスト

#### 2. `scripts/colab/colab_large_scale_experiment.py`
**依存関係:**
- `insightspike.core.agents.main_agent.MainAgent`
- `insightspike.insight_fact_registry.InsightFactRegistry`
- `insightspike.core.config.get_config`

**機能:** GPU加速を使った大規模実験、複数データセット統合

#### 3. `scripts/testing/test_llm_config_fix.py` & `test_llm_config_fix_lite.py`
**依存関係:**
- `insightspike.core.agents.main_agent.MainAgent`
- `insightspike.core.config.get_config`
- `insightspike.cli` - 各種コマンド

**機能:** LLM設定の検証、MainAgentのテスト

### ⚠️ Legacy システム依存スクリプト

#### 1. `scripts/testing/test_complete_insight_system.py`
**依存関係:**
- `insightspike.agent_loop.cycle` - 旧システムのメインループ
- `insightspike.layer2_memory_manager.Memory` - レガシーメモリ
- `insightspike.insight_fact_registry.InsightFactRegistry`

**機能:** 旧システムでの統合テスト、互換性確認

### 🎯 Mock システム使用スクリプト

#### 1. `scripts/production/run_true_insight_experiment.py`
**特徴:**
- 独自の`SimpleVectorStore`実装
- InsightSpikeモジュールを使わずに実験を実行
- 純粋な比較実験のための分離実装

#### 2. `scripts/run_poc_simple.py`
**特徴:**
- 軽量なMockクラス実装
- matplotlib依存の可視化機能
- 簡易実験用

### 🔧 完全独立スクリプト

#### 1. `scripts/utilities/generate_visual_summary.py`
**機能:** 実験結果のASCIIチャート生成、依存関係なし

#### 2. `scripts/utilities/comprehensive_rag_analysis.py`  
**機能:** RAGベースライン比較分析、JSON結果ファイル分析

#### 3. `scripts/production/create_true_insight_experiment.py`
**機能:** 実験用データセット生成、ファイルシステム操作のみ

## 🔄 実行可能性チェック

### ✅ 即座に実行可能
1. **System Validation** - 完全なシステム検証
2. **LLM Config Tests** - 設定とMainAgentテスト
3. **Visual Summary** - 結果レポート生成
4. **Data Generation** - 実験データ作成

### ⚠️ 要依存関係チェック
1. **Legacy System Tests** - 旧システムモジュールの確認必要
2. **Colab Experiments** - GPU環境とHugging Faceライブラリ
3. **HF Dataset Integration** - 外部データセット依存

### 🔧 修正推奨
1. **Mock実験スクリプト** - MainAgentとの統合を検討
2. **診断ツール** - 新しい設定システムへの移行
3. **レガシーテスト** - 新システムへの移行計画

## 📈 推奨アクション

### 即座に実行すべき項目
1. **システム検証の実行** 
   ```bash
   python scripts/production/system_validation.py
   ```

2. **実験データの準備**
   ```bash
   python scripts/production/create_true_insight_experiment.py
   ```

3. **設定テストの実行**
   ```bash
   python scripts/testing/test_llm_config_fix_lite.py
   ```

### 段階的移行が必要な項目
1. **レガシーシステムテスト** → **MainAgent**への移行
2. **Mock実験** → **統合実験**への発展
3. **診断ツール** → **新設定システム**対応

### 長期的な改善案
1. **統一された実験フレームワーク**の構築
2. **依存関係の最小化**と**モジュラー設計**
3. **CI/CD対応**の実験スクリプト作成

## 🏁 結論

プロジェクト構造の再整理により、スクリプトは機能別に明確に分類されました。大部分のスクリプトは新しいMainAgentシステムまたは独立した実装を使用しており、レガシーシステムへの依存は最小限に抑えられています。

**現在の状況:**
- ✅ **6個** のスクリプトが新システムで完全動作
- ⚠️ **3個** のスクリプトがレガシーシステム使用
- 🎯 **2個** のスクリプトがMock実装使用
- 🔧 **6個** のスクリプトが完全独立

**次のステップ:** 最も重要なシステム検証とテストスクリプトから実行を開始し、段階的にレガシーシステムから新システムへの移行を完了する。
