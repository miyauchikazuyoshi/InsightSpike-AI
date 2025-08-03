# 論文実験構成案

## 実験の流れ

### 1. 概念実証：迷路実験（2D/2.5D）
- **目的**: geDIGの基本原理を視覚的に示す
- **環境**: シンプルな2D迷路（MacBookで実行可能）
- **示すこと**:
  - 障害物（矛盾）を避ける経路探索
  - Wake-Sleepサイクルの効果
  - 洞察（ショートカット）の発見

### 2. 言語空間での検証：Sentence-BERT
- **目的**: 実用的なNLPタスクでの有効性
- **環境**: CPU実行可能（768次元）
- **実験内容**:
  - RAT（Remote Associates Test）
  - 質問応答での洞察生成
  - 距離ベース検索 vs コサイン類似度

### 3. マルチモーダル拡張：CLIP
- **目的**: より豊かな洞察空間の可能性
- **環境**: 要GPU（ただし工夫次第で...）
- **実験内容**:
  - テキスト→画像の連想
  - 視覚的メタファーの生成
  - クロスモーダル洞察

### 4. 将来展望：geDIG専用空間
- **ビジョン**: 洞察生成に最適化された埋め込み空間
- **特徴**:
  - 矛盾を明示的にエンコード
  - 洞察の「ひらめきやすさ」を距離に反映
  - Wake-Sleep学習による自己組織化

## CLIPの計算資源問題と対策

### 軽量化オプション

1. **CLIP-ViT-B/32（最小モデル）**
```python
# メモリ使用量比較
CLIP-ViT-L/14: ~1.7GB
CLIP-ViT-B/32: ~340MB（5分の1）
CLIP-ViT-B/16: ~350MB

# MacBookでも動く可能性
```

2. **事前計算戦略**
```python
# 実験データのみ事前にエンコード
embeddings = {}
for text in experiment_texts:
    embeddings[text] = clip.encode_text(text)
    
# 保存して再利用
np.save('clip_embeddings.npy', embeddings)
```

3. **量子化版の使用**
```python
# OpenCLIP の int8 量子化版
# メモリ使用量を1/4に削減
```

4. **CPU最適化**
```python
# OpenVINOやONNX Runtimeで高速化
import onnxruntime
clip_onnx = onnxruntime.InferenceSession("clip.onnx")
```

### 実験規模の調整

```python
# フルスケール実験（要GPU）
full_dataset = load_full_dataset()  # 1000サンプル

# 論文用小規模実験（CPU可）
demo_dataset = load_demo_dataset()  # 50サンプル
- 主要な現象は再現
- 計算時間は許容範囲
```

## 論文での記述例

### 実験設定
```
実験環境：
- 迷路実験: MacBook Pro (M1, 8GB RAM)
- Sentence-BERT: 同上
- CLIP実験: 
  - 小規模デモ: MacBook Pro
  - フルスケール: Google Colab (Tesla T4)
```

### 制限事項の明記
```
計算資源の制約により、CLIP実験は代表的な
50サンプルでの検証に留めた。しかし、これらの
結果は、より大規模な実験でも同様の傾向を示す
ことが期待される。
```

## 実装の優先順位

1. **必須（論文の核心）**
   - 迷路実験 ✓
   - Sentence-BERT実験 ✓
   - 基本的なgeDIG検証 ✓

2. **推奨（説得力向上）**
   - CLIP小規模デモ
   - 視覚的な結果例

3. **オプション（将来研究）**
   - 大規模CLIP実験
   - geDIG専用空間の設計

## 代替案：CLIP-Lite

もしCLIPが重すぎる場合：
1. **ImageBind-Small**: Metaの軽量マルチモーダル
2. **Japanese-CLIP-ViT-B-16**: 日本語特化で軽量
3. **概念図のみ**: CLIPの代わりに概念的な図解

この構成なら、MacBookでも説得力のある論文が書けそうです！