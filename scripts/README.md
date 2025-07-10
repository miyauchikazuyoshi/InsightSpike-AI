# Scripts Directory

## 📋 概要

InsightSpike-AIプロジェクトの各種スクリプト・ツール・ユーティリティを整理したディレクトリ

## 🔧 保守・管理スクリプト

### **プッシュ前バリデーション**

- `pre_push_validation.py` - **プッシュ前検証 + データ状態管理**
  - データ整合性チェック
  - クリーンバックアップからの自動復元
  - 実験バリデーション実行

  ```bash
  python scripts/pre_push_validation.py
  ```

### **データ管理**

- `utilities/restore_clean_data.py` - **データ復元・バックアップ**

  ```bash
  # クリーンデータに復元
  python scripts/utilities/restore_clean_data.py --restore
  
  # 新しいクリーンバックアップ作成
  python scripts/utilities/restore_clean_data.py --backup
  ```

### **Git統合**

- `git-hooks/pre-push` - **Git pre-pushフック**

  ```bash
  # インストール
  cp scripts/git-hooks/pre-push .git/hooks/pre-push
  chmod +x .git/hooks/pre-push
  ```

## 📁 ディレクトリ構成

| ディレクトリ | 説明 | 主要スクリプト |
|-------------|------|---------------|
| `debugging/` | 🔍 デバッグスクリプト | 状態診断・問題解析 |
| `testing/` | 🧪 テストスクリプト | コンポーネント・データテスト |
| `validation/` | ✅ 検証スクリプト | 品質保証・妥当性確認 |
| `production/` | 🚀 本番運用スクリプト | システム検証 |
| `colab/` | 📓 Colab統合スクリプト | Google Colab用ツール |
| `ci/` | ⚙️ CI/CD スクリプト | 自動化・継続統合 |

## 🚀 使用方法

```bash
# プロジェクトルートから実行
cd /path/to/InsightSpike-AI

# デバッグ
python scripts/debugging/debug_experiment_state.py

# テスト
python scripts/testing/safe_component_test.py

# 検証
python scripts/validation/validate_mermaid.py
```

## 🔧 その他のスクリプト

### **セットアップ**
- `setup_models.py` - 必要なモデルのダウンロード
- `setup_tinyllama.py` - TinyLlamaモデルのセットアップ
- `post_install.py` - インストール後の初期設定

### **ユーティリティ**
- `utilities/persistent_load_documents.py` - ドキュメントの永続的な読み込み

## 📊 整理完了実績

**2025年01月10日 更新**:
- ✅ 不要なスクリプト5個を削除
- ✅ CLIコマンドを新形式に更新 (spike query/embed)
- ✅ ドキュメントを最新状態に更新

**2024年07月01日 完了**:

- ✅ 不要なディレクトリ削除 (experiments/, organization/, utilities/)
- ✅ 古いファイル削除 (重複・非使用スクリプト)
- ✅ コア機能に集約された軽量な構造達成
- ✅ 保守・管理に必要なスクリプトのみ保持
- ✅ プッシュ前バリデーションにデータ管理機能統合

---

## InsightSpike-AI Project - Clean Scripts Structure
