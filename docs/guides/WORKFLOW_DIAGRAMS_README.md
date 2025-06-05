# InsightSpike-AI Workflow Visualization

このディレクトリには、InsightSpike-AIの現在のワークフローを可視化したMermaid図が含まれています。🌳

## 📊 Available Diagrams

### 1. 🌊 [WORKFLOW_TREE.mermaid](./WORKFLOW_TREE.mermaid)
**全体ワークフロー概観**
- ユーザーインターフェースから洞察活用までの完全な流れ
- 4層アーキテクチャの詳細
- フィードバックループと学習メカニズム
- CLI コマンドの統合

### 2. 🏗️ [TECHNICAL_ARCHITECTURE.mermaid](./TECHNICAL_ARCHITECTURE.mermaid)
**技術アーキテクチャ詳細**
- レイヤー別の技術実装
- データベースとストレージ構造
- 設定管理システム
- 各コンポーネント間の依存関係

### 3. 💡 [INSIGHT_LIFECYCLE.mermaid](./INSIGHT_LIFECYCLE.mermaid)
**洞察のライフサイクル**
- 洞察の発見から活用までの全過程
- 品質評価システムの詳細
- 登録判定フロー
- 継続的学習メカニズム

### 4. 🎛️ [SYSTEM_DASHBOARD.mermaid](./SYSTEM_DASHBOARD.mermaid)
**現在のシステム状態**
- コンポーネントの動作状況
- 洞察データベースの統計
- 設定状態とヘルスチェック
- 推奨されるNext Actions

## 🚀 How to View

### VS Code with Mermaid Extension
```bash
# Mermaid Preview拡張機能をインストール
code --install-extension bierner.markdown-mermaid
```

### Online Mermaid Editor
1. [Mermaid Live Editor](https://mermaid.live/) にアクセス
2. ファイルの内容をコピー&ペースト
3. リアルタイムでプレビュー

### CLI で画像生成
```bash
# Mermaid CLIをインストール
npm install -g @mermaid-js/mermaid-cli

# PNG画像として出力
mmdc -i WORKFLOW_TREE.mermaid -o workflow_tree.png
mmdc -i TECHNICAL_ARCHITECTURE.mermaid -o technical_architecture.png
mmdc -i INSIGHT_LIFECYCLE.mermaid -o insight_lifecycle.png
mmdc -i SYSTEM_DASHBOARD.mermaid -o system_dashboard.png
```

## 🧠 Key Insights from Visualization

### ✅ 完全に実装済み
- **洞察自動抽出**: 応答から自動的に洞察を発見
- **品質評価システム**: 多基準による洞察の評価
- **データベース永続化**: SQLiteによる構造化ストレージ
- **CLI管理インターフェース**: 直感的な洞察管理
- **フィードバックループ**: 学習による継続的改善

### ⚠️ 部分的実装・制限事項
- **PyTorchグラフ推論**: PyTorch/PyG未インストールによる制限
- **GPU利用**: 現在はCPUのみでの動作
- **大規模モデル**: 軽量モデルでの動作（TinyLlama）

### 🎯 脳らしい特徴
1. **記憶形成**: 経験から洞察を抽出・蓄積
2. **関連付け**: 概念間の関係を自動発見
3. **品質判定**: 重要な情報のみを保持
4. **継続学習**: 過去の洞察が新しい推論に影響
5. **適応的調整**: 使用パターンに基づく最適化

## 📈 Current System Status

```
🧠 Insight Registry Status: ACTIVE
📊 Total Insights: 6
⭐ Average Quality: 0.67
🔍 Search Index: READY
🔄 Learning Loop: OPERATIONAL
```

## 🎨 Diagram Color Coding

- 🔵 **Blue**: ユーザーインターフェース
- 🟣 **Purple**: 処理・計算コンポーネント
- 🟢 **Green**: ストレージ・データベース
- 🟠 **Orange**: 洞察関連システム
- 🔴 **Pink**: フィードバック・学習

これらの図により、InsightSpike-AIが単なるAIツールではなく、**学習し続ける人工的な脳** として機能していることが明確に可視化されています。🚀
