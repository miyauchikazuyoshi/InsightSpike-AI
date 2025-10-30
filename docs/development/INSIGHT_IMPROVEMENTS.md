---
status: active
category: insight
owner: refactor-wg
created: 2025-08-25
last_review: 2025-08-25
---

# InsightSpike改善案：IGとスパイク検出

## 1. 現状の問題認識

### 1.1 IGの解釈
- **現状**: IGが負の値を示す（エントロピー増加）
- **実は正しい**: 初期学習段階では知識が増えてばらつきが増加するのは自然
- **人間の認知**: 人間は自然に「意味のある構造化」として解釈し直す

### 1.2 質問処理の課題
```
人間の処理:
「微分と積分の関係は？」
→ 「微分とは何か」「積分とは何か」「関係性」に分解

AIの処理:
「微分と積分の関係は？」
→ 一塊として処理 → エントロピー飽和
```

## 2. 改善案

### 2.1 質問の自動リフレーミング

```python
class QuestionReframer:
    """質問を意味のあるチャンクに分解"""
    
    def reframe_question(self, question: str) -> List[str]:
        """
        質問を複数の観点に分解
        
        例: "微分と積分の関係は？"
        → ["微分とは何か", 
           "積分とは何か", 
           "微分と積分の逆操作性",
           "具体例での関係"]
        """
        # LLMを使って質問を分解
        prompts = [
            f"質問を学習しやすい小さな質問に分解: {question}",
            f"この質問の前提知識は何？: {question}",
            f"この質問の本質は何？: {question}"
        ]
        
        sub_questions = []
        for prompt in prompts:
            response = self.llm.query(prompt)
            sub_questions.extend(self.parse_sub_questions(response))
        
        return sub_questions
```

### 2.2 適応的チャンクサイズ

```python
class AdaptiveChunker:
    """エントロピーに基づくチャンクサイズ調整"""
    
    def chunk_knowledge(self, text: str, target_entropy: float = 0.7) -> List[str]:
        """
        テキストを適切なサイズのチャンクに分割
        
        - 短すぎる: 情報不足
        - 長すぎる: エントロピー飽和
        - 適切: エントロピーが target_entropy 付近
        """
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            # チャンクのエントロピーを計算
            chunk_text = " ".join(current_chunk)
            entropy = self.calculate_text_entropy(chunk_text)
            
            if entropy > target_entropy * 1.2:  # 飽和
                # 1文戻す
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [sentence]
            elif entropy > target_entropy * 0.8:  # 適切
                chunks.append(chunk_text)
                current_chunk = []
        
        return chunks
```

### 2.3 多段階収束アプローチ

```python
class MultiStageConvergence:
    """複数ループでの知識収束"""
    
    def process_with_convergence(self, question: str, max_loops: int = 5):
        """
        IGが安定するまで繰り返し処理
        
        1. 初期ループ: 高エントロピー（知識収集）
        2. 中期ループ: エントロピー減少（知識整理）
        3. 後期ループ: 低エントロピー（洞察形成）
        """
        ig_history = []
        
        for loop in range(max_loops):
            # 質問をリフレーム
            if loop == 0:
                sub_questions = self.reframer.reframe_question(question)
            else:
                # 前回の結果を踏まえて再リフレーム
                sub_questions = self.adaptive_reframe(question, ig_history)
            
            # 各サブ質問を処理
            loop_ig = 0
            for sub_q in sub_questions:
                result = self.agent.process_question(sub_q)
                loop_ig += result.metrics.get('delta_ig', 0)
            
            ig_history.append(loop_ig)
            
            # 収束判定
            if self.is_converged(ig_history):
                break
        
        return ig_history
    
    def is_converged(self, ig_history: List[float]) -> bool:
        """収束判定: IGの変化が小さくなったら"""
        if len(ig_history) < 2:
            return False
        
        # 最近2つのIGの差が小さい
        recent_change = abs(ig_history[-1] - ig_history[-2])
        return recent_change < 0.1
```

### 2.4 新しいスパイク検出基準

```python
class ImprovedSpikeDetection:
    """改良されたスパイク検出"""
    
    def detect_spike(self, metrics: Dict) -> bool:
        """
        複数の基準でスパイク検出
        
        1. 従来: GED < -0.1 and IG > 0.05
        2. 新規: 
           - IGの符号反転（負→正）
           - IGの急激な変化（|ΔIG| > 0.5）
           - GEDとIGの相関（構造化と理解の同期）
        """
        ged = metrics.get('delta_ged', 0)
        ig = metrics.get('delta_ig', 0)
        
        # 従来の基準
        traditional = ged < -0.1 and ig > 0.05
        
        # IGの履歴から符号反転を検出
        ig_history = metrics.get('ig_history', [])
        sign_flip = False
        if len(ig_history) >= 2:
            sign_flip = ig_history[-2] < 0 and ig_history[-1] > 0
        
        # IGの急激な変化
        ig_jump = abs(ig) > 0.5
        
        # GEDとIGの相関（両方が改善方向）
        correlation = (ged < 0 and ig > -0.1) or (ged < -0.3)
        
        return traditional or sign_flip or (ig_jump and correlation)
```

## 3. 実装優先順位

1. **Phase 1**: 質問リフレーミング
   - 最も効果が高い
   - 既存システムへの影響が小さい

2. **Phase 2**: 適応的チャンクサイズ
   - エントロピー飽和を防ぐ
   - 学習効率の向上

3. **Phase 3**: 多段階収束
   - より深い洞察の獲得
   - 計算コストとのバランス

4. **Phase 4**: スパイク検出改良
   - より多様な洞察パターンの検出

## 4. 期待される効果

1. **学習の自然さ**: 人間の認知プロセスに近い学習
2. **エントロピー管理**: 適切な情報量での学習
3. **洞察の質向上**: 表面的でない深い理解
4. **スパイク検出率向上**: より多くの「あは体験」を捕捉