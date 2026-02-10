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
| ★★ | Gao et al. (2025) "Weight-sparse transformers have interpretable circuits" | β₁削減と解釈性の関係（Pruning Paradox） | [arXiv](https://arxiv.org/abs/2511.13653) |
| ★★ | Oyama et al. (2025) "Mapping 1,000+ Language Models via the Log-Likelihood Vector" | H空間でのモデルマップ（F拡張の基盤） | [ACL Anthology](https://aclanthology.org/2025.acl-long.1584/) |

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

### Hewitt & Manning (2019) — 構文構造の埋め込み発見

**引用：**

> Hewitt, J. & Manning, C. D. (2019).
> A Structural Probe for Finding Syntax in Word Representations.
> In Proceedings of NAACL-HLT 2019, pp. 4129–4138.
> [PDF](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf) | [Code](https://github.com/john-hewitt/structural-probes)

**彼らの貢献：**

hidden state に線形変換 B を適用すると、変換後の L2 距離が構文木上のホップ距離に、
ノルムが構文木の深さに対応するという発見。
すなわち **「構文木構造が hidden state 内に線形的に埋め込まれている」** ことの実証。

**本研究での使い方：**

Hewitt & Manning は **各層の静的なスナップショット** として B 行列を提供する。
本研究はこの B 行列を「測定器」として借用し、以下の **層間動態の量** を新たに定義した:

| 量 | 定義 | Hewitt & Manning にない点 |
|----|------|--------------------------|
| EPC(l) | ‖D(l) − D(l−1)‖_F / ‖D(l−1)‖_F | 層間の距離行列**変化率**。彼らは各層の距離行列を個別に評価したが、層間差分は取っていない |
| SP(l) | SpearmanCorr(depth(l), depth(l−1)) | 深さ予測の層間**安定度**。彼らは各層での深さ再現精度を報告したが、層間相関は定義していない |
| Δβ₁(l) | β₁(l) − β₁(l−1) | 距離グラフのトポロジー変化。Hewitt & Manning の枠組みに存在しない量 |

つまり、Hewitt & Manning は **「何が埋め込まれているか」** を発見し、
本研究は **「それが層を跨いでどう変化するか」** を F 分解として定式化した。
B 行列は借用するが、EPC・SP・β₁ の層間動態としての定義と、
それらを F = ΔEPC − λ(ΔH + γΔβ₁) に統合する構成は本研究の独自の知的貢献である。

**借りるもの：** 線形変換 B（距離 probe、深さ probe）、訓練コード
**新規：** B による測定結果の層間差分を EPC/SP として定義し、geDIG F 分解に統合する構成

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

## 設計強化ポイント（追加提案）

本節は、既存設計を壊さずに「普遍性主張（同一(λ,γ)を固定した係数転移でも複数モデルでR²が高い）」を強化するための追加仕様。

### 1. 事前登録（過剰適合の抑制）
- [ ] Grid search実行前に `results/preregister_hypotheses.yaml` を保存し、以下を固定する  
  - 主要主張: 「同一(λ,γ)で全モデルR² > 0.7」  
  - 補助主張: 「negative controlでR²が有意に低下」  
  - 解析範囲: 使用モデル、使用データ、除外条件、評価指標
- [ ] 事後に閾値を変更した場合は、`post_hoc` と明記して別セクションで報告する

### 2. 統計強度の明示（点推定だけで終わらせない）
- [ ] 各モデルのR², slopeに対して95%信頼区間を付与（文単位bootstrap推奨）
- [ ] baseline/controls差分に対し、効果量（Cliff's delta か Cohen's d）を併記
- [ ] レポートに `n_sentences`, `n_layers`, `n_valid_points` を必須で出力

### 3. 係数転移の厳格化（本命）
- [ ] `Leave-One-Model-Out` を追加  
  - 例: BERT系で推定した(λ*,γ*)をGPT系へ固定転移、逆方向も実施
- [ ] 転移性能を行列で可視化（train-model × test-model のR²表）
- [ ] 「モデル別最適」と「固定係数」の差分を `ΔR²` で明示

### 4. コントロールの拡張（反証耐性）
- [ ] 既存の `λ=0, γ=0` に加え、以下のdropout型アブレーションを追加  
  - `F = ΔEPC - λΔH`（SP項なし）  
  - `F = ΔEPC - λγΔSP`（H項なし）  
  - `F = -(ΔH + γΔSP)`（EPC項なし）
- [ ] 層順シャッフル（layer index permutation）で線形降下が崩れることを確認
- [ ] ランダム直交行列 `B_random` を使った擬似SP/EPCで偽陽性率を測定

### 5. RoPE対応の分岐計画（Gemma等）
- [ ] RoPE系モデルを本検証に入れる前に、適用可否ゲートを明記  
  - 条件A: probe再現指標（UUAS/DSpr）が最小水準を満たす  
  - 条件B: SP/EPCの層変化が非退化（全層ほぼ定数でない）
- [ ] 条件未達の場合は「RoPEは別系統検証」として主張スコープを明確に限定

### 6. 外部教師依存の縮小（次段の布石）
- [ ] `B_supervised`（構文木教師あり）に加え、`B_self`（位置符号部分空間/PCA由来）の比較実験を追加
- [ ] 比較指標: `corr(SP_supervised, SP_self)`, `ΔR²(F軌跡)`, 係数転移の頑健性
- [ ] 結果に応じて、将来計画を二分  
  - 高整合: 教師なし近似へ移行  
  - 低整合: まずは教師あり版を「測定器」として固定

### 7. 失敗条件の明文化（主張の自己規律）
- [ ] 固定(λ,γ)転移で `R² > 0.7` を満たすモデルが過半未満なら、普遍性主張は「保留」
- [ ] negative controlとbaselineの95%CIが重なる場合、因果主張は弱めて記述
- [ ] RoPE未検証のままなら、結論を「additive positional系で確認」に限定

---

## 追加考察と予測（2026-02）

本節は、直近の実行結果（BERT/GPT-2系でのβ1採用実験）とディスカッションを踏まえ、
「何を主張するか」を過不足なく整理するための補助設計である。

### A. 考察：普遍なのは係数そのものより関係式

先行結果では、モデルごとの最適 `(λ, γ)` が一致しない一方で、同一モデル内での再現性は比較的高い。
この事実は、次のように解釈するのが妥当である。

1. 生の `(λ, γ)` は、`ΔEPC, ΔH, Δβ1` のスケール差を吸収する「単位変換係数」の性質を持つ
2. したがって、異なるアーキテクチャ間で生の係数一致を最初の成功条件に置くと過剰に厳しい
3. 一方で、最適化後の `F` の形状（線形性、層遷移パターン、controlとの差）は比較可能である

このため主張軸を以下に再定義する。

- 旧主張（強すぎる初期仮説）: 同一 `(λ, γ)` が全モデルで高R²
- 新主張（検証可能）: 最適化後 `F` の構造指標がモデル品質と相関する

### B. 検証仮説（事前登録候補）

#### H1: 学習効果仮説（random初期化との差）

同一モデルで、

$$
\Delta R^2_{learn} = R^2(F_{baseline}) - R^2(F_{random\_init})
$$

を定義する。高品質モデルほど `ΔR²_learn` が大きくなることを予測する。

#### H2: 構造保全仮説（入力破壊との差）

同一モデルで、

$$
\Delta R^2_{struct} = R^2(F_{baseline}) - \max(R^2(F_{shuffle}), R^2(F_{random\_init}))
$$

を定義する。学習済み構造を活用できるモデルほど `ΔR²_struct` が大きくなることを予測する。

#### H3: 係数収束仮説（同一系列内）

同一ファミリ（BERT系列、GPT系列）内でパラメータ規模が増えるほど、
最適 `(λ, γ)` の分散が減少する（系列内収束）。

### C. 予測（モデル系列ごとの順序予測）

数値の一点予測ではなく、順序予測として事前に定義する。

#### BERT系列（同一系列のスケール比較）

予測順序（`ΔR²_struct` の小→大）:

`bert-tiny < distilbert-base-uncased < bert-base-uncased < bert-large-uncased`

#### GPT系列（同一系列のスケール比較）

予測順序（`ΔR²_struct` の小→大）:

`sshleifer/tiny-gpt2 < distilgpt2 < gpt2 < gpt2-medium`

#### 係数に関する予測

1. 生の `(λ, γ)` は BERT系列とGPT系列で一致しない
2. 各項を標準化した後（z-score化）に再推定すると、系列内の係数分散は縮小する
3. 高品質モデルほど、`baseline` における層プロファイル（`ΔEPC, ΔH, Δβ1`）の再現性が高い

### D. 代替説明と識別方針

`random_init` で R² が高くなる現象は、次の2解釈を区別する必要がある。

1. 均一変換仮説: 未学習モデルは各層が均質で、見かけ上直線になりやすい
2. 学習効率仮説: 高品質モデルほど情報処理が効率化し、より直線になる

識別方針として、`R²` 単独ではなく `ΔR²_learn`, `ΔR²_struct`, 層曲率（2次差分）を併用する。

### E. 反証条件（この節の自己規律）

以下のいずれかを満たした場合、本節の主張は棄却または保留する。

1. 同一系列で `ΔR²_struct` の順序予測が過半で成立しない
2. `ΔR²_learn` がモデル品質（標準ベンチマーク指標）と無相関
3. 係数分散がスケール増加で縮小しない（系列内収束が見られない）

---

## 関連研究と接続仮説（2026-02 追加）

本節では、F 分解の位置づけを明確にするため、直近の関連研究2件との接続を整理する。

### 関連研究 1: Weight-Sparse Transformers and Interpretable Circuits

> Gao, L., Rajaram, A., Coxon, J., Govande, S. V., Baker, B. & Mossing, D. (2025).
> Weight-sparse transformers have interpretable circuits.
> arXiv:2511.13653.
> [arXiv](https://arxiv.org/abs/2511.13653)

**概要:**
Transformer の重みをスパース化（大部分をゼロ）すると、各ニューロンの接続が少数に限定され、
人間が解釈可能な回路（circuit）が発見しやすくなる。
ただし capability と interpretability の間にはトレードオフが存在し、
モデル規模を拡大するとこのトレードオフが改善される。

**F 分解との接続:**

この発見は、matchstick figure で可視化した **Pruning Paradox** の実例と位置づけられる。

| Gao et al. の知見 | F 分解での解釈 |
|-------------------|---------------|
| 重みスパース化 → 解釈性向上 | 辺（接続）の削減 → β₁ 低下 → F 改善 |
| capability-interpretability トレードオフ | ΔEPC の損失 vs Δβ₁ の削減のバランス |
| スケール拡大でトレードオフ改善 | 大規模モデルほど EPC を保ちつつ β₁ を削減できる |

具体的には:

1. **スパース化 = β₁ 削減操作**。ニューロン間の接続を刈ると、表現空間の距離グラフのループ（β₁）が減る。
   Gao et al. は「各ニューロンが少数の接続しか持たない」ことが解釈性の鍵だと述べているが、
   これは F の第3項 Δβ₁ の削減に他ならない。

2. **capability の維持 = EPC の保全**。スパース化しても入力-出力の距離構造が保たれていれば、
   モデルの表現能力は維持される。capability が落ちるのは EPC が劣化した場合。

3. **F による定量化の可能性**。現在 Gao et al. はスパース化の度合いと capability/interpretability を
   個別に報告しているが、F = ΔEPC − λ(ΔH + γΔβ₁) の枠組みで統一的に記述すれば、
   「どの程度のスパース化が最適か」を F の最小化問題として定式化できる可能性がある。

**検証可能な予測:**
- スパースモデルの各層 F 軌跡は、対応する dense モデルよりΔβ₁ が小さく、F の線形性が高い
- capability 低下が顕著なスパースモデルでは、ΔEPC が劣化（構造変化が粗くなる）している

### 関連研究 2: Mapping 1,000+ Language Models via the Log-Likelihood Vector

> Oyama, M., Yamagiwa, H., Takase, Y. & Shimodaira, H. (2025).
> Mapping 1,000+ Language Models via the Log-Likelihood Vector.
> In Proceedings of ACL 2025, Volume 1: Long Papers, pp. 32983–33038.
> [ACL Anthology](https://aclanthology.org/2025.acl-long.1584/)
> **ACL 2025 Outstanding Paper Award.**

**概要:**
対数尤度ベクトル（固定コーパスに対する各モデルの log-likelihood 値）をモデルの座標とし、
1000以上の言語モデルを幾何空間にマッピングする。
二乗ユークリッド距離が KL divergence を近似するという理論的保証を持ち、
計算コストはモデル数・サンプル数に対して線形。

**F 分解との接続:**

Oyama et al. のモデルマップは、**F の H 項のみによるモデル空間の構成**と解釈できる。

| 空間の次元 | Oyama et al. | F 分解 |
|-----------|-------------|--------|
| 情報量（確率） | log-likelihood ≈ H | ΔH |
| 構造（距離） | — | ΔEPC |
| トポロジー（ループ） | — | Δβ₁ |

接続の要点:

1. **H 空間からの拡張**。Oyama et al. は H（対数尤度）の1次元でモデルを配置し、
   それだけで Outstanding Paper を獲得した。
   F 分解は、ここに EPC（構造変化）と β₁（トポロジー変化）の2軸を追加する。
   これにより、H だけでは区別できない「同程度の perplexity だが内部構造が異なるモデル」を分離できる。

2. **モデルマップ上の F 等高線**。Oyama et al. のマップ上で、各モデルの F 軌跡指標
   （R², ΔR²_learn, ΔR²_struct）を色やサイズで可視化すれば、
   モデル品質と F の関係が直接観察可能になる。
   具体的には、マップ上で高品質モデルが集まる領域で F の線形性が高い（R² が大きい）
   ことが予測される。

3. **KL divergence と F の関係**。Oyama et al. はモデル間距離 ≈ KL divergence を示した。
   KL divergence はエントロピーの差分（H 項）で定義されるが、
   F は KL に構造項（EPC, β₁）を加えたものと見なせる。
   つまり F は **構造を考慮した拡張 KL divergence** としての解釈を持つ。

**検証可能な予測:**
- Oyama et al. のマップで近傍に位置するモデル（H が類似）でも、F 軌跡パターンは異なる
- マップ上のモデル品質（ベンチマークスコア）と F の ΔR²_struct が正の相関を示す
- 高品質モデルのクラスタでは、最適 (λ, γ) の分散が小さい（§C の H3 仮説の可視化）

### 3研究の統合的位置づけ

```
Oyama et al. (2025)          Gao et al. (2025)
   H（エントロピー）空間で         スパース化（β₁削減）で
   モデルを配置                   解釈性を向上
        │                            │
        │    ┌─────────────────┐      │
        └───→│ F = ΔEPC        │←─────┘
             │   − λ(ΔH        │
             │   + γΔβ₁)       │
             └────────┬────────┘
                      │
              3項の統一的記述:
              ・H = 確率的不確実性（Oyama et al.）
              ・EPC = 構造変化コスト
              ・β₁ = トポロジー的複雑性（Gao et al.）
```

- Oyama et al. は F の「H 射影」でモデル空間を記述した
- Gao et al. は F の「β₁ 射影」で解釈性を改善した
- 本実験は、3項すべてを用いた統一的記述により、
  モデル品質・解釈性・表現力を同一の枠組みで説明することを目指す

### スケーリング仮説（今後の課題: 大規模モデルでの検証）

現在の実験対象（BERT-base, GPT-2, DistilBERT 等）は比較的小規模であり、
最適 (λ, γ) のモデル間不一致はアーキテクチャの未成熟さに起因する可能性がある。

**仮説:** 十分に大規模かつ高品質なモデル（Llama 3.1 70B+, DeepSeek-V3 等）では、
内部の情報処理が洗練されているため、3項の分離がクリーンになり、
(λ, γ) の系列間分散が縮小する。

物理学のアナロジーで言えば、小規模モデルは「高温の気体」（分子運動がランダム）、
大規模高品質モデルは「結晶」（規則構造が可視化可能）に対応する。
結晶構造は十分に低温でなければ観察できないように、
F の普遍性も十分に高品質なモデルでなければ検出できない可能性がある。

**⚠️ 現時点の制約:**
本研究は個人環境で実施しており、大規模モデル（70B+）の hidden state 抽出には
GPU メモリ・計算時間の制約がある。
現段階では小〜中規模モデル（〜1B）での検証を完了させ、
大規模モデルでの検証は今後の課題として位置づける。

**今後の検証候補:**

| モデル | パラメータ | 層数 | 計算要件 |
|--------|-----------|------|---------|
| Llama 3.1 8B | 8B | 32 | GPU 16GB（4bit量子化時） |
| Llama 3.1 70B | 70B | 80 | A100 80GB × 1-2 |
| Qwen2.5-72B | 72B | 80 | A100 80GB × 1-2 |
| DeepSeek-V3 | 671B (MoE) | 61 | マルチ GPU 必須 |

同一アーキテクチャ内のスケール比較（Llama 8B → 70B）が最もクリーンな検証となる。
4bit 量子化での hidden state 取得が精度に与える影響の事前検証も必要。

---

## 実行進捗と追加検証計画（2026-02-10時点）

### 現状進捗（multi-model 実測）

実測結果ディレクトリ:

- `experiments/transformer/inference_gedig_v2/results/transfer_beta1_multi64_6models_20260210T194406`
- 集計:
  - all-track: `multi_model_metrics.csv`, `multi_model_metrics.md`
  - token-only: `multi_model_metrics_token_lm.csv`, `multi_model_metrics_token_lm.md`

本実験の主張に使う **primary track は token_lm のみ**。
Sentence-transformers は定義差（構造・H・EPC）を伴うため secondary/exploratory として分離する。

Primary (token_lm) の有効モデル数は `8`（総数 `9`, invalid `1`）。

| model | baseline R² | shuffle R² | random R² | ΔR²_learn | ΔR²_struct | baseline best (λ,γ) |
|---|---:|---:|---:|---:|---:|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.0290 | 0.0030 | 0.0910 | -0.0620 | -0.0620 | (10.0, 10.0) |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | 0.0199 | 0.0047 | 0.0898 | -0.0699 | -0.0699 | (1.0, 10.0) |
| bert-base-uncased | 0.0087 | 0.0260 | 0.0192 | -0.0105 | -0.0173 | (0.01, 0.01) |
| distilbert-base-uncased | 0.3535 | 0.3733 | 0.7727 | -0.4192 | -0.4192 | (0.01, 0.01) |
| distilbert-base-uncased-finetuned-sst-2-english | 0.3309 | 0.2912 | 0.8175 | -0.4866 | -0.4866 | (10.0, 10.0) |
| distilgpt2 | 0.0188 | 0.0358 | 0.7962 | -0.7773 | -0.7773 | (0.01, 0.01) |
| gpt2 | 0.0808 | 0.0883 | 0.1673 | -0.0864 | -0.0864 | (0.5, 10.0) |
| gpt2-medium | 0.0741 | 0.0536 | 0.1389 | -0.0648 | -0.0648 | (0.01, 0.01) |

Secondary (sentence_embedding) は exploratory:

- all-MiniLM-L6-v2
- all-MiniLM-L12-v2
- all-mpnet-base-v2
- paraphrase-MiniLM-L6-v2

### 判断（現時点）

1. 仮説を即却下する段階ではない（モデル数・系列カバレッジが不足）
2. ただし現状データだけでは仮説整合性を支持できない（primary token_lm の `ΔR²_struct`, `ΔR²_learn` は全て負）
3. 当面の焦点は「random_init優位の原因切り分け」と「系列内順位の安定検証」
4. GPT系の部分順位は仮説と整合（`distilgpt2 < gpt2 < gpt2-medium`）
5. BERT系は `distilbert-base-uncased-finetuned-sst-2-english` が悪化し、タスク特化重みの影響を疑う（primary内でも要注意）

### 中間考察（論拠と傾向の整理）

#### 1. 現状データから読める「成立方向の傾向」

primary token_lm では `ΔR²_struct`, `ΔR²_learn` が全て負であり、
現時点で仮説成立を主張することはできない。
ただし、系列内では次の改善方向が観測される。

- GPT系列で `ΔR²_struct` が単調改善:
  `distilgpt2 (-0.7773) < gpt2 (-0.0864) < gpt2-medium (-0.0648)`
- 同時に `random_r2` は単調低下:
  `0.7962 -> 0.1673 -> 0.1389`
- TinyLlama系列でも改善方向:
  `intermediate-step-1431k-3T (-0.0699) < chat-v1.0 (-0.0620)`

このパターンは、「モデルが未成熟なほど random-init 優位が強く、
成熟に伴って baseline 側へ寄る」という解釈と整合する。

#### 2. 関連研究との接続（リソース要請の論拠）

- Oyama et al. (2025) は、LLV空間でのモデル間距離（KL近似）が能力差の説明に有効であることを示した。
  本実験の結果は、現サンプルがその空間で比較的「未成熟側」に偏っている可能性を示唆する。
- Entropy-Lens (2025) は、同系列内でサイズを跨ぐ整列的な層動態を報告しており、
  本実験の系列内改善（GPT/TinyLlama）はその方向に一致する初期兆候と見なせる。
- Gao et al. (2025) は、解釈性と能力のトレードオフがスケールで改善することを示す。
  本実験でも、小規模側ほど `random_r2` が高い傾向が見られ、
  「規模拡張で構造項が効く」仮説の動機づけになる。

#### 3. 中間結論（本段階の主張可能範囲）

本段階で主張できるのは次の水準である。

1. 仮説の直接支持は未達（符号は未反転）
2. ただし系列内の改善傾向は観測され、未成熟モデル仮説とは整合
3. 仮説の判定には大規模モデル検証が必須であり、追加計算資源の投入は合理的

#### 4. 追加資源を用いた最小検証セット（優先）

- 同一系列スケール比較: `8B -> 70B`（可能なら中間チェックポイントも）
- 指標: `ΔR²_struct`, `ΔR²_learn`, `random_r2`, 最適 `(λ,γ)` 分散
- 判定:
  - 期待方向: スケール増加で `ΔR²_struct` は単調増加（負から0近傍へ）
  - 反証方向: スケールを上げても `ΔR²_struct` が改善しない

### 追加検証方法（次ラウンド）

#### V1. モデル数拡張（最優先）

- 目標: 有効モデル `>=10`（最低 `8`）
- ルール: `baseline mean_fit.r2` が算出できるモデルのみ有効
- まずはローカルキャッシュ済み safetensors モデルから追加し、実行失敗率を下げる
- 進捗:
  - all-track 有効 `12` 到達
  - primary token_lm は有効 `8` 到達（旧目標 `>=8` を満たした）

#### V2. 系列内順位検証（H2/H3 直結）

- BERT系とGPT系を分離し、`ΔR²_struct` の順位一致率を算出
- 判定: 事前順位予測の過半一致で「暫定整合」、未満は保留

#### V3. 係数比較の正規化

- 生の `(λ, γ)` に加え、各項 `z-score` 正規化後の `(λ, γ)` を再推定
- 目的: スケール差由来の見かけの不一致を分離
- 判定: 正規化後に系列内分散が縮小するかを確認

#### V4. 形状指標の追加（R²単独依存を回避）

- 層曲率（2次差分）と層転移点（急変層）を算出
- `random_init` 高R²が「均一変換」由来かを識別

#### V5. 固定係数転移の再評価

- 各モデル最適 vs 固定 `(λ,γ)`（共通候補）で `ΔR²_transfer` を比較
- leave-one-model-out で汎化性能を評価

### 進行ルール（運用）

- `post_hoc` 指標を増やす場合は、一次指標（H1/H2/H3）と分離して報告
- 実験停止条件:
  - 有効モデル `>=10` かつ 系列内順位が過半不一致
  - または `ΔR²_struct` が全系列で一貫して負のまま改善しない

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
