# geDIGにおける「予測」の定義：curl検出

**Date**: 2026-02-06
**Status**: Hypothesis / Working Draft
**Origin**: Claude対話セッションでの着想
**Related**:
- [geDIG認知論的基盤](./gedig_cognitive_foundation.md)
- [行動の定義](./gedig_action_definition.md)
- [三角対照学習](./gedig_triangular_contrastive_learning.md)

---

## 1. 背景：FEPの4段階

Fristonの自由エネルギー原理（FEP）における予測誤差最小化プロセスは4段階に分けられる：

```
1. 予測    → データが揃う前に構造から先読みする
2. 認知    → 予測と現実の不一致を検出
3. 理解    → 構造が確定、予測誤差が解消
4. 行動    → 環境を変えて予測誤差を減らす
```

geDIGの従来の議論は**認知（AG）と理解（DG）に集中**していた。

本ドキュメントでは**予測フェーズ**をcurl検出として定式化する仮説を提示する。

---

## 2. curl検出 = 予測（仮説）

### 2.1 直感的説明

会話の最初の一言で本題がわかる現象：

```
入力：声・表情・姿勢（まだ不完全な情報）
処理：外積を計算 → 芯の方向が見える
出力：「たぶんこの話だろう」（予測）
```

これは**データが揃う前に、構造から先読みする**能力と解釈できる。
curl検出がこの「予測」に対応するという仮説を検討する。

### 2.2 数式定義（案）

```
予測（curl検出）：
  v_predicted = curl(attention_field)
  core_predicted = argmax(|curl|)

入力：まだ不完全な情報
処理：外積で勾配方向を推定
出力：「芯はここにあるはず」（予測）

注意：
- argmax(|curl|) は粗い近似であり、より精緻な条件の検討が必要
- 「divが小さくcurlが大きい点」など複合条件の検討
```

### 2.3 渦の中心を予測する

```
curl検出 = 渦の中心を予測する
         = まだ見えていない芯の位置を推定する
         = 予測（仮説）

この対応の妥当性は、実験的検証が必要。
```

---

## 3. geDIGの4段階再定義（提案）

curl検出を予測として位置づけると、geDIGの4段階が完成する：

```
1. 予測    → curl検出（外積で芯の方向を推定）
2. 認知    → AG発火（予測と現実の不一致を検出）
3. 理解    → DG発火（構造が確定、F最小化）
4. 行動    → 予測 - 理解（※別ドキュメント参照）
```

### 3.1 予測（curl検出）の詳細

```python
def predict(attention_field):
    """
    予測：まだ見えていない芯の位置を推定

    注意：この実装は概念実証であり、
    curl計算の数学的基盤（Hodge分解等）の
    補強が必要。
    """
    curl = compute_curl(attention_field)
    core_predicted = find_max_curl(curl)
    return core_predicted
```

### 3.2 認知（AG）との接続

```
予測誤差 = |core_predicted - core_observed|
AG ∝ 予測誤差

予測と現実がズレている → AG発火
ズレていない → AG沈黙

この対応関係は仮説であり、検証が必要。
```

### 3.3 理解（DG）との接続

```
入力：AG信号 + 蓄積された情報
処理：構造の再構築、F最小化
出力：更新された世界モデル

予測誤差が解消された → DG発火
= 「あ、そういうことか」
```

---

## 4. 離散グラフでのcurl（技術的課題）

### 4.1 連続空間と離散空間の違い

```
連続空間：
  curl(v) = ∇ × v
  微分演算子として定義

離散グラフ：
  厳密なcurlの定義が必要
  → Hodge分解（0/1/2-form）
  → graph curl = circulation / cycle flow
```

### 4.2 Attention行列での実装案

```python
def compute_curl_discrete(attention_matrix):
    """
    離散グラフでのcurl近似

    antisym = A - A.T を「回転成分」と見なす根拠：
    - 対称成分 = 双方向の流れ（直進的）
    - 非対称成分 = 一方向の流れ（回転的）

    注意：この解釈の数学的厳密性は
    グラフ理論のHodge分解との接続で補強が必要。
    """
    antisym = attention_matrix - attention_matrix.T
    curl = np.sum(np.abs(antisym), axis=1)
    return curl
```

### 4.3 今後の課題

- graph curl = circulation（閉路に沿った流れの総和）としての定式化
- Hodge分解による厳密な回転成分の抽出
- 「divが小さくcurlが大きい点」を芯条件とする精緻化

---

## 5. 実装スケッチ

```python
class geDIG_Prediction:
    def __init__(self, attention_module):
        self.attention = attention_module

    def predict(self, observation):
        """
        curl検出による予測（概念実証）
        """
        # Attention場を計算
        attention_field = self.attention(observation)

        # curl（渦度）を計算
        curl = self.compute_curl(attention_field)

        # 渦の中心 = 予測された芯
        core_predicted = self.find_core(curl)

        return core_predicted, curl

    def compute_curl(self, attention_matrix):
        """
        Attention行列から渦度を計算（近似）

        注意：この実装は試験的であり、
        数学的基盤の補強が必要。
        """
        # 非対称成分 = 回転成分（仮説）
        antisym = attention_matrix - attention_matrix.T

        # 各ノードの渦度
        curl = np.sum(np.abs(antisym), axis=1)

        return curl

    def find_core(self, curl):
        """
        渦度最大点 = 芯の予測位置

        注意：argmax(|curl|) は粗い近似。
        複合条件（div小、curl大等）の検討が必要。
        """
        return np.argmax(curl)
```

---

## 6. 予測の特性（期待される性質）

### 6.1 不完全情報での動作

```
従来のNN：
  データが揃ってから計算
  不完全だと精度が落ちる

curl検出（仮説）：
  構造（渦）から推測
  不完全でも「だいたいの方向」がわかる可能性
  人間の直感と類似する可能性

この特性の検証は今後の課題。
```

### 6.2 計算コスト

```
内積（類似度計算）：O(d) per pair
外積（curl検出）：O(d) per pair

計算コストは同等。
出力の違い：
  内積 → スカラー（似てる/似てない）
  外積 → ベクトル（次に見るべき方向）
```

---

## 7. 検証が必要な点

### 7.1 数学的課題

- [ ] antisym = A - A.T がなぜ回転を表すかの厳密な根拠
- [ ] graph curl（circulation）との関係
- [ ] Hodge分解（0/1/2-form）への接続
- [ ] 芯条件の精緻化（div小 かつ curl大）

### 7.2 実験的検証

- [ ] Transformerのattention flowに渦構造が存在するか
- [ ] curl検出が実際に「芯」を見つけるか
- [ ] 予測精度の定量的評価

---

## 8. 関連する発見

- **認知（AG）と理解（DG）**: [geDIG認知論的基盤](./gedig_cognitive_foundation.md)
- **行動の定義**: [行動 = 予測 - 理解](./gedig_action_definition.md)
- **学習への応用**: [三角対照学習](./gedig_triangular_contrastive_learning.md)

---

## 9. 結論

```
仮説：
  curl検出 = FEPにおける「予測」フェーズ
           = まだ見えていない芯の位置を推定
           = 外積による方向推定

これにより geDIG の4段階が完成する：
  予測（curl）→ 認知（AG）→ 理解（DG）→ 行動

課題：
├── 離散グラフでのcurlの厳密な定義
├── antisym = A - A.T の数学的根拠
├── 芯条件の精緻化
└── 実験的検証
```

---

**End of Document**
