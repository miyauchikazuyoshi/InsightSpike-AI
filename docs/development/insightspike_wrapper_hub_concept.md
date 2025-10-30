---
status: proposal
category: insight
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
decision_due: 2025-09-15
---

# InsightSpike ラッパーアプリケーション構想

## 概要
InsightSpikeをローカルで動作するアプリケーションとしてラップし、Gemini/Claudeなどの外部LLMへのインテリジェントなハブとして機能させる。ユーザーの会話から閃きや洞察を抽出し、それをプロンプトとして活用する。

## コアコンセプト
「思考の第二の脳」- ユーザーの思考を拡張し、深め、記録するパーソナルAIアシスタント

## アーキテクチャ構想

### 1. ローカルファースト設計
```
User Input → InsightSpike (Local) → Enhanced Prompt → External LLM → Response
                    ↓
              Knowledge Base
              (SQLite + Graph)
```

### 2. インテリジェントハブ機能
- **入力処理**: ユーザーの質問/思考を受け取る
- **ローカル処理**: 
  - SQLiteDataStoreに知識として保存
  - GEDIGで関連する過去の知識を検索
  - 洞察スパイクを検出
- **プロンプト強化**: 
  - ユーザーの元の質問
  - 検出された洞察
  - 関連する過去の文脈
  を組み合わせて高品質なプロンプトを生成
- **外部LLM連携**: 強化されたプロンプトを外部LLMに送信
- **応答統合**: 外部LLMの応答と洞察を統合してユーザーに返す

### 3. 洞察の継続的学習
```python
# 概念的な実装イメージ
class InsightHubApp:
    def __init__(self):
        self.agent = DataStoreMainAgent(SQLiteDataStore(), config)
        self.external_llms = {
            'claude': AnthropicProvider(),
            'gemini': GeminiProvider(),
            'gpt4': OpenAIProvider()
        }
    
    def process_thought(self, user_input: str, target_llm: str = 'claude'):
        # 1. 知識として保存
        self.agent.add_knowledge(user_input)
        
        # 2. ローカルで洞察を抽出
        insight = self.agent.process(user_input)
        
        # 3. プロンプトを強化
        enhanced_prompt = self.create_enhanced_prompt(
            original=user_input,
            insights=insight.get('spike_info'),
            context=insight.get('related_knowledge')
        )
        
        # 4. 外部LLMに問い合わせ
        response = self.external_llms[target_llm].generate(enhanced_prompt)
        
        # 5. 結果を統合
        return {
            'response': response,
            'insights': insight,
            'has_spike': insight.get('has_spike', False)
        }
```

## UI実装オプション

### 1. CLI拡張版
- 現在の`spike`コマンドを拡張
- インタラクティブモードで会話を継続
- シンプルで軽量

### 2. Web UI (Streamlit/Gradio)
```python
# Streamlitでの実装例
import streamlit as st

st.title("🧠 InsightSpike - Your Second Brain")

# チャット履歴
if "messages" not in st.session_state:
    st.session_state.messages = []

# ユーザー入力
user_input = st.chat_input("あなたの思考を入力...")

if user_input:
    # InsightSpikeで処理
    result = hub.process_thought(user_input)
    
    # 洞察スパイクの表示
    if result['has_spike']:
        st.balloons()
        st.info(f"💡 洞察を検出！{result['insights']['spike_info']['summary']}")
    
    # 応答表示
    st.chat_message("assistant").write(result['response'])
```

### 3. デスクトップアプリ (Tauri/Electron)
- ネイティブアプリとして配布
- システムトレイ常駐
- ホットキーでクイックアクセス

## 実装ロードマップ

### Phase 1: コア機能実装
- [ ] 外部LLMプロバイダーの追加（Gemini）
- [ ] プロンプト強化エンジンの実装
- [ ] 会話履歴の永続化と検索

### Phase 2: UI開発
- [ ] Streamlitプロトタイプ
- [ ] チャット形式のインターフェース
- [ ] 洞察の可視化（グラフ表示）

### Phase 3: 高度な機能
- [ ] マルチモーダル対応（画像、音声）
- [ ] 自動要約とレポート生成
- [ ] プラグインシステム

## 技術的な利点

1. **完全なプライバシー**: すべての思考履歴はローカルに保存
2. **オフライン対応**: コア機能はインターネット不要
3. **コスト効率**: 重要な質問のみ外部LLMを使用
4. **カスタマイズ性**: 個人の思考パターンに適応

## 期待される使用例

### 1. 日常的な思考の記録
```
User: "今日のミーティングで量子コンピューティングの話が出た。面白かったけど、生物学との関連性があるのかな？"
InsightSpike: [知識として保存]
→ 1週間後
User: "DNAの情報処理について調べてる"
InsightSpike: "💡 洞察スパイク！量子コンピューティングとDNA情報処理の類似性を検出しました"
```

### 2. アイデアの熟成
```
User: "新しいアプリのアイデア..."（毎日少しずつ入力）
InsightSpike: 関連するアイデアを自動的に結びつけ、新しい視点を提供
```

### 3. 学習の深化
```
User: 学んだことを自分の言葉で記録
InsightSpike: 概念間の関連性を発見し、より深い理解を促進
```

## 実装上の課題と解決策

### 1. レスポンス速度
- **課題**: ローカル処理 + 外部API = 遅延
- **解決**: 非同期処理、キャッシング、段階的な応答表示

### 2. コスト管理
- **課題**: 外部LLM APIのコスト
- **解決**: 
  - 重要度に応じてLLMを選択
  - ローカルLLMのフォールバック
  - 月間予算設定機能

### 3. データ管理
- **課題**: 長期使用でのデータ肥大化
- **解決**: 
  - 自動アーカイブ機能
  - 重要度ベースの圧縮
  - エクスポート/インポート機能

## まとめ

このラッパーアプリケーションは、InsightSpikeの真の価値を引き出す最終形態です。単なるQ&Aツールではなく、ユーザーの思考を拡張し、新しい洞察を生み出すパートナーとして機能します。

現在のアーキテクチャは、この構想を実現するために必要なすべての要素を既に備えています。あとは、適切なUIでラップし、ユーザー体験を洗練させるだけです。