# geDIG Transformer推論実験設計書 v2

## 実験目的

Transformerの推論時（固定モデルにおける層ごとのベクトル変換）において、geDIG F値が線形降下することを検証する。

迷路実験（エージェントの学習過程）で確認されたgeDIG F分解：

$$
F^{(l)} = \Delta EPC^{(l)} - \lambda(\Delta H^{(l)} + \gamma \Delta SP^{(l)})
$$

が、Transformerの推論（層を通過するhidden stateの変換過程）にも適用可能であり、
**モデル非依存の係数λ,γで複数アーキテクチャにわたりF値が線形降下する**ことを示す。

---

## リソース・リンク集

### 必読論文

| 優先度 | 論文 | geDIGでの用途 | リンク |
|--------|------|---------------|--------|
| ★★★ | Hewitt & Manning (2019) "A Structural Probe for Finding Syntax in Word Representations" | EPC, SPの操作的定義の根拠 | [PDF](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf) / [ACL Anthology](https://aclanthology.org/N19-1419/) |
| ★★★ | Ali et al. (2025) "Entropy-Lens: The Information Signature of Transformer Computations" | Hの操作的定義の根拠 | [arXiv](https://arxiv.org/abs/2502.16570) / [HTML版](https://arxiv.org/html/2502.16570) |
| ★★ | Ethayarajh (2019) "How Contextual are Contextualized Word Representations?" | 補助的引用（異方性・文脈特異性） | [ACL Anthology](https://aclanthology.org/D19-1006/) |

### コード・データ

| リソース | リンク | 備考 |
|----------|--------|------|
| structural-probes リポジトリ | [GitHub](https://github.com/john-hewitt/structural-probes) | B行列の訓練コード一式。距離probe・深さprobeの両方を含む |
| Hewitt ブログ記事（図解付き） | [Blog](https://www.cs.columbia.edu/~johnhew//structural-probe.html) | 論文より先に読むべき。直感的な理解に最適 |
| Penn Treebank (PTB) 本体 | [LDC](https://catalog.ldc.upenn.edu/LDC99T42) | **有料ライセンス（LDC会員制）**。初期段階では不要 |
| English UD (EWT) データ | structural-probes リポジトリ同梱 | `bash scripts/download_example.sh` で自動取得。**PTBなしでprobe訓練・検証が可能** |

### 読む順序（推奨）

```
1. Hewitt ブログ記事         → 直感をつかむ                        10分
2. Hewitt & Manning 論文PDF  → B行列の訓練手順・評価指標を理解      30分
3. structural-probes README  → 実行手順・yaml設定の把握             20分
4. Entropy-Lens 論文         → H定義確認・モデル別プロファイル把握   20分
```

### ⚠️ セットアップ時の既知の問題

| 問題 | 詳細 | 対処 |
|------|------|------|
| GPT-2のactivationが巨大 | 数百単位の値が出るためprobeの最適化が失敗する | probe行列のinitを小さくする（README FAQ参照） |
| `pytorch-pretrained-bert` が旧版 | リポジトリは旧パッケージに依存 | `transformers` への置き換えが必要な場合あり |
| PTBライセンスが有料 | LDC会員でなければ即座に取得不可 | **EWTデータで先にパイプライン全体を検証**してからPTBを検討 |
| RoPEモデル（Gemma等） | Structural Probeは加法的位置符号化を前提 | Phase 1でBERT/GPT-2を先に検証し、RoPEモデルは結果次第で別途検討 |

---

## 先行実験からの教訓

### 問題点

前回の推論実験（`experiments/transformer/inference_f_trajectory/`）では、
EPC/H/SPの定義が推論時の物理量として不十分であり、以下の問題が生じた：

| 問題 | 詳細 |
|------|------|
| bathtub curve | F値がU字型（0.56→0.19→0.47）、線形降下せず |
| entropy_sign | エントロピーの符号を事後的に選択するパラメータが必要だった |
| SP = 1.00 | Causal LMでSPが常に1.0（CLSトークンが存在しないため） |
| ネガティブコントロール失敗 | ランダムモデルでもF降下が観測された |

### 根本原因

定義が「既存のパーツを組み合わせた」ものであり、推論時のhidden stateから自然に導出されていなかった。
特に、attentionエントロピー（メカニズム側）とhidden state（ベクトル側）の混在が問題。

### 本実験での解決

先行研究で検証済みの2つの手法を、geDIGのF分解の各項に対応させる：

- **Structural Probe** (Hewitt & Manning, 2019) → EPC, SP
- **Entropy-Lens** (Ali et al., 2025) → H

---

## 各項の操作的定義

### H（確率側：語彙エントロピー）

**根拠：** Entropy-Lens (Ali et al., 2025)

各層のhidden stateを、モデルの出力デコーダ（unembedding行列）で語彙空間に射影し、
softmax後のShannon entropyを計算する。

$$
H^{(l)} = -\sum_{v \in V} p(v | h^{(l)}) \log p(v | h^{(l)})
$$

ここで：

$$
p(v | h^{(l)}) = \text{softmax}(h^{(l)} W_{\text{unembed}}^T)_v
$$

**先行研究での確認済みパターン：**

- GPT: 高 → 単調減少 → 低
- Gemma: 低 → 高 → 低（ベルカーブ）
- Llama: 低 → 高(維持) → 低（台形型）

これはhidden state自身が「次トークン予測としてどれだけ確定しているか」の直接的測定。
attentionエントロピー（メカニズム側の測定）より原理的に適切。

**geDIGでの解釈：**

- H高 = 確率的に不確実（溶融状態：多くの可能性が開いている）
- H低 = 確率的に確定（結晶状態：特定の出力に収束）

**ΔHの定義：**

$$
\Delta H^{(l)} = H^{(l)} - H^{(l-1)}
$$

### EPC（構造側：木距離行列の変化）

**根拠：** Structural Probe (Hewitt & Manning, 2019)

線形変換行列Bを学習し、変換後の空間で
hidden state間のL2距離が構文木上のホップ数に対応するようにする。

$$
d_B(h_i^{(l)}, h_j^{(l)})^2 = (B(h_i^{(l)} - h_j^{(l)}))^T (B(h_i^{(l)} - h_j^{(l)}))
$$

$$
d_B(h_i^{(l)}, h_j^{(l)})^2 \approx d_{\text{tree}}(w_i, w_j)
$$

**各層で得られる距離行列 $D^{(l)}$** を、層間で比較する：

$$
EPC^{(l)} = \frac{\| D^{(l)} - D^{(l-1)} \|_F}{\| D^{(l-1)} \|_F}
$$

ここで $\| \cdot \|_F$ はFrobenius norm。

**geDIGでの解釈：**

- EPC高 = 構造の大規模な組み替え（辺の編集コスト大）
- EPC低 = 構造が安定（変化少ない）

**迷路との対応：** 迷路のEPCはグラフ上の辺編集距離。推論時のEPCは、B変換後の距離行列の変化量。
同一の線形変換Bから両方の量が計算される点が、定義の統一性を保証する。

### SP（構造側：木の深さの安定度）

**根拠：** Structural Probe (Hewitt & Manning, 2019) - depth probe

同じ線形変換Bにより、hidden stateのノルムが構文木の深さ（根からの距離）に対応する：

$$
\| B h_i^{(l)} \|^2 \approx \text{depth}(w_i)
$$

SPは、この深さ構造が層間でどれだけ安定しているかを測る：

$$
SP^{(l)} = \text{SpearmanCorr}(\text{depth\_pred}^{(l)}, \text{depth\_pred}^{(l-1)})
$$

**geDIGでの解釈：**

- SP高 = 木構造（ハブ-リーフ関係）が安定（ショートカットが機能）
- SP低 = 木構造が不安定（階層が組み替え中）

**迷路との対応：** 迷路のSPはスタートからゴールへの最短経路効率。
推論時のSPは、根（ハブ）から各ノード（トークン）への階層的距離の安定度。

### 注意：Bの学習について

Structural Probeの行列Bは、外部の構文解析木（Penn Treebank等）を教師データとして学習する。
これはgeDIGの「自己完結性」に対する制約となるが、以下の理由で許容する：

1. Bは**一度だけ**学習すれば、推論時には固定パラメータとして使える
2. Hewitt & Manning (2019) により、Bの有効ランクは64-128程度と極めて低次元
3. 将来的にはBを位置符号の部分空間から導出する理論的経路がある（後述）

---

## 実験手順

### Phase 1: Structural Probeの訓練

**目的：** 線形変換行列Bを取得する

**手順：**

1. English UD (EWT) データを使用（`bash scripts/download_example.sh` で取得済み）
   - PTBは有料ライセンスのため、まずEWTでパイプライン全体を検証
   - 必要に応じてPTBライセンスを後日取得
2. 各モデルの各層からhidden stateを抽出
3. Hewitt & Manning (2019) の公開コード (https://github.com/john-hewitt/structural-probes) を使用
4. 距離probe（木距離を再現するB_dist）と深さprobe（木の深さを再現するB_depth）を別々に訓練
5. 各モデル × 各層でBを取得

**対象モデル：**

| モデル | 層数 | 隠れ次元 | 位置符号化 | 役割 |
|--------|------|----------|-----------|------|
| BERT-base-cased | 12 | 768 | Learned (additive) | 主実験・λγ決定 |
| BERT-large-cased | 24 | 1024 | Learned (additive) | 同ファミリー検証 |
| GPT-2 small | 12 | 768 | Learned (additive) | Causal LM検証 |
| GPT-2 medium | 24 | 1024 | スケール検証 |
| Gemma-2B* | 18 | 2048 | RoPE | 異アーキテクチャ |

*Gemmaは位置符号がRoPEのため、Structural Probeの適用可否を事前確認する必要がある。
RoPEモデルでは回転操作が暗黙的に組み込まれており、加法的分解が直接適用できない可能性がある。
Phase 1の結果次第で、RoPEモデルは別途検討とする。

**⚠️ GPT-2の注意：** activationの値が非常に大きい（数百単位）。probe行列の初期値を小さくしないとスコアが出ない。
[structural-probes README FAQ](https://github.com/john-hewitt/structural-probes#readme) に対処法の記載あり。

### Phase 2: geDIG各項の計測

**目的：** 各モデルの各層でH, EPC, SPを計算する

**手順：**

1. テストセットの文（EWT test + 追加の自然文）を各モデルに入力
2. 各層のhidden stateを保存
3. 以下を計算：

```python
for layer in range(num_layers):
    # H: 語彙エントロピー (Entropy-Lens)
    logits = hidden_states[layer] @ model.lm_head.weight.T
    probs = softmax(logits, dim=-1)
    H[layer] = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
    
    # EPC: B変換後の距離行列の変化
    D[layer] = pairwise_distance(hidden_states[layer] @ B_dist.T)
    if layer > 0:
        EPC[layer] = frobenius_norm(D[layer] - D[layer-1]) / frobenius_norm(D[layer-1])
    
    # SP: 深さ予測の安定度
    depth_pred[layer] = torch.norm(hidden_states[layer] @ B_depth.T, dim=-1) ** 2
    if layer > 0:
        SP[layer] = spearman_corr(depth_pred[layer], depth_pred[layer-1])
```

4. 差分を計算：
```python
delta_H[layer]   = H[layer] - H[layer-1]
delta_EPC[layer] = EPC[layer] - EPC[layer-1]
delta_SP[layer]  = SP[layer] - SP[layer-1]
```

### Phase 3: λ, γの決定とF軌跡の評価

**目的：** BERT-baseでλ,γを最適化し、他モデルに転移

**手順：**

**Step 1: BERT-baseでのgrid search**

```
λ ∈ [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
γ ∈ [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
```

各(λ, γ)について：
```python
F[layer] = delta_EPC[layer] - λ * (delta_H[layer] + γ * delta_SP[layer])
R2 = linear_regression_R2(layers, F)
```

R²が最大となる(λ*, γ*)を選択。

**Step 2: 係数の転移検証**

λ*, γ*を**固定のまま**、BERT-large, GPT-2 small, GPT-2 mediumに適用。
各モデルのR²を報告。

**成功基準：**

| レベル | 条件 | 意味 |
|--------|------|------|
| 最低限 | BERT-baseでR² > 0.8 | 定義が概ね正しい |
| 中程度 | 全モデルでR² > 0.7（モデル別λγ） | F線形降下は普遍的だが係数はモデル依存 |
| 強い | 全モデルでR² > 0.7（同一λγ） | F分解の構造自体が普遍的 |
| 最強 | R² > 0.9（同一λγ）+ λγに理論的導出 | 理論的に閉じた体系 |

### Phase 4: コントロール実験

**ネガティブコントロール：**

1. **ランダム初期化モデル** — 学習していないモデルでF軌跡を計測。
   線形降下しない（or R²が大幅に低い）ことを確認。

2. **シャッフルされた入力** — 正しく学習したモデルに、トークン順をランダムにした入力を与える。
   構文木が破壊されるため、EPC/SPの動態が変化することを確認。

3. **λ=0, γ=0** — F = ΔEPCのみ。線形降下しないことを確認し、
   H項とSP項の寄与が本質的であることを示す。

**ポジティブコントロール：**

4. **構文的に複雑な文 vs 単純な文** — 
   複雑な文ほど「溶融→再結晶」のダイナミクスが顕著になるか検証。

### Phase 5: 追加分析（ultrametricity）

**目的：** geDIG Fとは独立に、木構造の出現を検証する

各層のB変換後の距離行列に対し、ultrametricityを計算：

$$
\text{ultrametricity}^{(l)} = 1 - \frac{1}{|T|} \sum_{(i,j,k) \in T} \mathbb{1}[d(i,k) > \max(d(i,j), d(j,k))]
$$

ここでTは全トリプルの集合。

**予測：**

- 仮説A（構造的再構成）：深い層でultrametricity上昇 → Fの深層での挙動と相関
- 仮説B（確率的先鋭化）：ultrametricityは層によらず低いまま

この分析はF分解とは独立だが、「なぜFが降下するか」の因果的説明を与える。

---

## 先行研究の引用方法

### Hewitt & Manning (2019) — EPC, SPの根拠

**引用：**

> Hewitt, J. & Manning, C. D. (2019).
> A Structural Probe for Finding Syntax in Word Representations.
> In Proceedings of NAACL-HLT 2019, pp. 4129–4138.
> [PDF](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf) | [Code](https://github.com/john-hewitt/structural-probes)

**使い方：**

本研究では、Hewitt & Manning (2019) のstructural probeが発見した
「線形変換後のL2距離が構文木距離に対応する」という性質を、
geDIGのEPC（構造変化コスト）とSP（階層安定度）の操作的定義として採用する。

彼らの貢献は**木構造が埋め込まれていることの発見**であり、
本研究の貢献は**その木構造の層間変化をgeDIGのF分解として定式化し、
複数モデルでの線形降下を示す**ことにある。

**借りるもの：** 線形変換B、距離probe、深さprobe、公開コード
**新規：** 層間の距離行列変化（EPC）と深さ安定度（SP）をgeDIG F分解に組み込む点

### Ali et al. (2025) — Hの根拠

**引用：**

> Ali, R., Caso, F., Irwin, C. & Liò, P. (2025).
> Entropy-Lens: The Information Signature of Transformer Computations.
> arXiv:2502.16570.
> [arXiv](https://arxiv.org/abs/2502.16570) | [HTML](https://arxiv.org/html/2502.16570)

**使い方：**

本研究では、Ali et al. (2025) のEntropy-Lensが発見した
「中間層のlogit-lens語彙エントロピーがモデルファミリー固有のプロファイルを持つ」
という知見を、geDIGのH（確率的不確実性）の操作的定義として採用する。

彼らの貢献は**エントロピープロファイルがモデルの計算シグネチャであることの発見**であり、
本研究の貢献は**そのエントロピーをF分解のH項として位置づけ、
構造変化（EPC, SP）との定量的関係を明らかにする**ことにある。

**借りるもの：** logit-lens語彙エントロピーの定義、モデルファミリー別プロファイルの知見
**新規：** エントロピーを構造変化と結合してF値を構成する点

### Ethayarajh (2019) — 補助的引用

**引用：**

> Ethayarajh, K. (2019).
> How Contextual are Contextualized Word Representations?
> In Proceedings of EMNLP-IJCNLP 2019, pp. 55–65.
> [ACL Anthology](https://aclanthology.org/D19-1006/)

**使い方：**

hidden stateの幾何的性質（異方性、文脈特異性）が層ごとに変化するという
先行知見として引用。特に「上層ほど文脈特異的」という発見は、
geDIGの「浅い層で溶融、深い層で再結晶」と整合する補助的証拠。

---

## 想定される批判と対応

### 批判1：「λ, γは恣意的パラメータ」

**対応：**

1モデルでフィッティングした係数が他モデルに転移することが実証的反論。
係数がモデル非依存であれば、それはF分解の構造が普遍的であることを意味する。
恣意的に見えるパラメータが実は普遍定数であるという発見は、
物理学における無次元定数（例：微細構造定数）と同型の議論である。

### 批判2：「Structural Probeは外部教師が必要」

**対応：**

現時点ではこれは正当な制約。ただし：
- Bの有効ランクが64-128と低いことから、構造情報は低次元部分空間に局在
- 将来的には位置符号の部分空間からBを理論的に導出する経路がある
- 本実験の主目的は「F線形降下の検証」であり、Bの自己完結的導出は次の課題

### 批判3：「線形降下は特定の文にのみ成立するのでは」

**対応：**

Phase 4のコントロール実験で対応。
- 複数の文長・構文複雑度で検証
- シャッフル入力で構文破壊時の挙動を確認
- 複数データセットでの再現性を報告

### 批判4：「logit-lensはtied embeddingを前提としている」

**対応：**

BERT/GPT-2はtied embedding。
non-tied embeddingのモデルでは、各層ごとに訓練されたaffine変換を使う
tuned-lens (Belrose et al., 2023) への拡張が自然な次ステップ。

---

## 実装ロードマップ

### Step 0: 環境構築（1日）
- [ ] structural-probes リポジトリのクローンと動作確認
  ```bash
  git clone https://github.com/john-hewitt/structural-probes/
  cd structural-probes
  conda install --file requirements.txt
  pip install pytorch-pretrained-bert
  ```
- [ ] EWTデータの取得（PTBは不要）
  ```bash
  bash scripts/download_example.sh
  # → example/data/ に train/dev/test の conll ファイルが生成される
  ```
- [ ] デモ実行で動作確認
  ```bash
  printf "The chef that went to the stores was out of food" | \
    python structural-probes/run_demo.py example/demo-bert.yaml
  ```
- [ ] 各モデルのhidden state抽出パイプライン構築

### Step 1: Structural Probe訓練（2-3日）
- [ ] BERT-base-cased の全12層で B_dist, B_depth を訓練
- [ ] BERT-large-cased の全24層で同様
- [ ] GPT-2 small/medium で同様
  - ⚠️ **GPT-2はprobe行列のinitを小さくすること**（README FAQ参照）
- [ ] 各probeのUUAS, DSpr, NSprを報告（Hewitt & Manningの結果と比較）

### Step 2: geDIG計測（1-2日）
- [ ] H（語彙エントロピー）の全層計測
- [ ] EPC（距離行列変化）の全層計測
- [ ] SP（深さ安定度）の全層計測
- [ ] 各量の層ごとプロットを生成

### Step 3: F軌跡分析（1日）
- [ ] BERT-baseでλ,γのgrid search
- [ ] 最適(λ*, γ*)でのF軌跡プロット
- [ ] 他モデルへの転移検証
- [ ] R²の報告

### Step 4: コントロール実験（1-2日）
- [ ] ランダムモデルでの計測
- [ ] シャッフル入力での計測
- [ ] λ=0, γ=0での計測
- [ ] 文の複雑度による比較

### Step 5: Ultrametricity分析（1日）
- [ ] 全層の距離行列からultrametricityを計算
- [ ] F軌跡との相関分析

**総所要時間：7-10日**

---

## 成功した場合のインパクト

### geDIG理論への寄与
- 迷路（離散グラフ）で発見されたF分解が、連続ベクトル空間（Transformer hidden state）にも適用可能であることの実証
- 「学習過程」だけでなく「推論過程」にもgeDIGが適用可能であることの証明
- F線形降下がTransformerの情報処理の普遍的性質であるという主張の根拠

### Transformer解釈性への寄与
- Hewitt & Manningの木構造（静的な各層のスナップショット）を、層間の動態として統一的に記述
- Entropy-Lensのエントロピープロファイル（確率側のみ）と構造変化（EPC/SP）の定量的関係を初めて示す
- 「溶融→再結晶」の相転移メタファーに定量的裏付けを与える

### 生成文法との接続
- Transformerが線形的入力から木構造を再構成する過程を、geDIGのF分解で定量化
- 人間の言語処理との構造的類似性に対する新しい定量的証拠
