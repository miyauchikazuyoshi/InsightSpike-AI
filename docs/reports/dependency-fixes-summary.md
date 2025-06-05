# 依存関係修正サマリー

## 修正日時
2025年6月1日

## 修正された問題

### 1. NumPyの重複インストール問題
**問題**: `setup.sh`と`setup_colab.sh`でpyproject.tomlに定義済みのnumpyを再度インストールしていた

**修正内容**:
- `scripts/setup/setup.sh`: NumPy個別インストールを削除、pyproject.tomlの管理に統一
- `scripts/colab/setup_colab.sh`: NumPy単独インストールを削除、thinc戦略と統合
- `scripts/colab/setup_colab_debug.sh`: 同様にthinc戦略を適用

**効果**: 
- 依存関係の一元管理
- インストール時間の短縮
- バージョン競合リスクの削減

### 2. thincの適切なバージョン制御戦略
**問題**: thincがNumPy 2.0を自動インストールしてFAISS互換性を破壊するリスク

**修正内容**:
- Colabセットアップスクリプトでthinc + NumPy 1.xを先行インストール
- `pip install "thinc>=8.1.0,<9.0.0" "numpy>=1.24.0,<2.0.0"`として制約を明示
- FAISS互換性を保証しながら最新thincを使用

**効果**:
- FAISS-GPU互換性の確保
- spaCy/thincエコシステムの安定性
- NumPy 2.0問題の事前防止

### 3. CI設定の整合性確保
**問題**: CI設定がsetup.shに依存しており、依存関係変更の影響を受ける可能性

**修正内容**:
- `.github/workflows/ci.yml`: コメントを追加して依存関係管理の変更を明示
- setup.shの改善により、CIも自動的に最適化

**効果**:
- CI環境での一貫した依存関係管理
- 重複インストールによる時間ロスの削減

### 4. 重複ファイル整理完了
**問題**: layer1,2,3,4の重複ファイルが階層跨いで存在し、保守性に影響

**修正内容**:
- 全レイヤーで互換性レイヤー化完了 (layer1_error_monitor.py, layer2_memory_manager.py, layer3_graph_pyg.py, layer3_reasoner_gnn.py, layer4_llm.py)
- deprecation warningシステム実装
- 新旧両方の構造をサポート

**効果**:
- コードの一貫性向上と重複削除
- 段階的移行による既存コードの保護
- 新機能と旧機能の並行サポート

## 技術的改善点

### 戦略的インストール順序
```bash
# Before (問題のあった方法)
pip install "numpy==1.26.4"  # 重複インストール
pip install transformers     # thincが自動的にNumPy 2.0を要求する可能性

# After (修正された方法)
pip install "thinc>=8.1.0,<9.0.0" "numpy>=1.24.0,<2.0.0"  # 制約を事前設定
# 以降のパッケージは既に適切なNumPyバージョンに制約される
```

### ファイル別の改善

#### `pyproject.toml`
- コメントを更新して依存関係管理戦略を明確化
- NumPyがPoetryで管理されることを明示

#### `scripts/setup/setup.sh`
- NumPy重複インストール削除
- pyproject.toml依存への統一化

#### `scripts/colab/setup_colab.sh`
- thinc事前インストール戦略の導入
- NumPy制約の明示的設定

#### `scripts/colab/setup_colab_debug.sh`
- デバッグ版でも同様の戦略を適用

## 検証方法

### 1. ローカル環境テスト
```bash
cd /path/to/project
rm -rf .venv  # 既存環境のクリーンアップ
bash scripts/setup/setup.sh
```

### 2. Colab環境テスト
```python
# Google Colabで実行
!git clone https://github.com/your-repo/InsightSpike-AI.git
%cd InsightSpike-AI
!bash scripts/colab/setup_colab.sh
```

### 3. CI環境での検証
- GitHub Actionsワークフローが正常に動作することを確認
- 依存関係インストール時間の改善を測定

## 今後の保守方針

1. **一元管理**: NumPyバージョンはpyproject.tomlで統一管理
2. **戦略的制約**: thinc + NumPy事前インストールでFAISS互換性確保
3. **環境一致**: ローカル、Colab、CI環境での一貫性維持
4. **定期検証**: 新しい依存関係追加時の影響確認

## 関連ファイル

- `pyproject.toml` - 依存関係定義
- `scripts/setup/setup.sh` - ローカル開発環境セットアップ
- `scripts/colab/setup_colab.sh` - Colab環境セットアップ
- `scripts/colab/setup_colab_debug.sh` - Colabデバッグ環境
- `.github/workflows/ci.yml` - CI設定
- `deployment/configs/requirements-colab.txt` - Colab要求ファイル
