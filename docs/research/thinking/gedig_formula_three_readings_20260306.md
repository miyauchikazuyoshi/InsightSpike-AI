# geDIG式の3つの読み替え

> **式の位置づけ（簡約式） / Formula Status (Simplified)**: この文書の数式は説明用の簡約式です。正準定義（Canonical）は `docs/gedig_spec.md` です。

**Date**: 2026-03-06  
**Status**: Interpretation Memo  
**Origin**: `gedig_origin_story.md` を正準としつつ、同一式の読み替え可能性を整理するためのメモ

**Related**:
- `docs/research/gedig_origin_story.md`
- `docs/research/thinking/betti_number_adoption_memo.md`
- `docs/research/thinking/betti1_engineering_spec.md`
- `docs/research/thinking/gedig_as_discrete_fep_schrodinger_analogy_20260227.md`
- `docs/paper/arxiv_v6_en/geDIG_onegauge_improved_v6_en.tex`

---

## 1. このメモの目的

geDIG の出発点は `origin story` にある

```text
F = cost - gain
```

という直感であり、これは現在でも最も重要な正準読解である。  
一方で、式を少し抽象化すると、同じ骨格から複数の理論的読解が見えてくる。

このメモの目的は、

1. 正準読解を固定すること
2. その上で二次的な読み替えを整理すること
3. 論文・実装・講演でどの読解を使うべきかを切り分けること

にある。

---

## 2. ここで使う簡約形

現在の canonical は、実装・論文上は

```text
F = ΔEPC - λ(ΔH + γ·SP)
```

である。  
このメモでは、読み替えを見やすくするため、係数を吸収した簡約形

```text
F~ = EPC - (H + B)
```

を使う。

ここで:

- `EPC`: 構造変更コスト
- `H`: 不確実性・未整理性・エントロピー項
- `B`: Betti side の位相項。現時点では実質 `β₁`（第一Betti数、独立サイクル数）

と読む。  
`B` は canonical symbol ではなく、このメモで `β₁` を短く書くための shorthand である。  
generic な「bridge gain」ではない。採用メモに従えば、`B` の中身は

- 位相数
- 穴の数
- 独立サイクル数
- 経路選択自由度

としての `β₁` である。

---

## 3. 読み 1: `EPC - (H + B)` = コスト - 利得（経済学的）

### 3.1 これが正準

最も自然で、`origin story` に忠実な読み方。

```text
EPC           = この更新はどれだけ構造コストを要するか
H + B         = その更新はどれだけ不確実性を減らし、位相数を改善するか
F が小さい   = コストに見合う更新
```

採用メモの言い方に合わせると、これは

```text
EPC - (ΔH + β₁)
```

の経済学的読解であり、

- これだけ編集して
- 情報がこれだけ整理され
- 位相的自由度がこれだけ増えた

を一つの収支として見る形である。  
これは geDIG の設計意図

- 探索するか
- 受理するか
- 保留するか
- 圧縮するか

をそのまま記述する。

### 3.2 この読みが強い理由

- `origin story` と整合する
- AG/DG 実装に直接落ちる
- Maze / RAG / OCR / Transformer すべてで使える
- 最も検証可能で、最も工学的

### 3.3 使用方針

この読みを **canonical** とする。  
論文本文、spec、実装説明ではこの読解を主軸に置く。

---

## 4. 読み 2: `(EPC - B) - H` = 構造 - エントロピー（物理学的）

### 4.1 何が見えるか

`B` を `β₁` として読むと、

```text
EPC - B   = 純粋な構造側の収支
H         = なお残る未整理性・エントロピー
```

となる。

採用メモに寄せると、

```text
EPC - β₁ = 冗長サイクルを除いた本質的構造変化
```

となる。  
これは

- 構造の安定化
- 骨格の変化
- サイクル冗長性の増減
- 秩序化と散逸

の釣り合いとして読める。

### 4.2 Helmholtz-like な読み

この読解では geDIG は

```text
構造収支 - エントロピー
```

という形になり、Helmholtz 自由エネルギーに似た

```text
骨格変化と散らばりの釣り合い
```

の式として読める。

重要なのは、これは **literal な熱力学同値** ではなく、**operational correspondence** だということ。

### 4.3 使用方針

- FEP / MDL / Helmholtz の discussion
- appendix
- 理論的エッセイ

には有効。  
ただし canonical として前面に出すと、実装との距離が広がる。

---

## 5. 読み 3: `(EPC - H) - B` = 内部状態 - 位相（生物学的）

### 5.1 何が見えるか

`EPC - H` を「情報的に説明しきれない構造変化」と読み、`B` を位相秩序・order parameter 側に置くと、

```text
内部状態の再編成 - 位相的秩序化
```

という形になる。

採用メモの言い方に合わせれば、

```text
EPC - ΔH = 情報的に説明できない構造変化
β₁       = その残差を説明する位相秩序
```

となる。  
この読みでは、`B` は単なる feature ではなく、

- 穴が生まれるか
- サイクルが固定化されるか
- ルーティング相が切り替わるか

を決める **相の指標** に近い。

### 5.2 相転移っぽい読み

この見方をすると、geDIG は

- 高エントロピーな探索相
- 構造が固定された秩序相
- その境界で起こるレジーム変化

を記述する式として見えてくる。

特に `β₁` を秩序パラメータと見る Landau 的な読解とは、この読みが最も近い。

### 5.3 使用方針

- phase transition 的な discussion
- β1 / routing / topology の役割整理
- speculative note

には有効。  
ただし、現時点では **最も遠い読解** であり、論文本文の主軸に据えるべきではない。

---

## 6. 3つの読みの関係

| 読み | 形 | 主な意味 | 使う場所 |
|---|---|---|---|
| 1 | `EPC - (H + B)` | コスト - 利得（経済学的） | canonical / 実装 / 本文 |
| 2 | `(EPC - B) - H` | 構造 - エントロピー（物理学的） | FEP/MDL/Helmholtz 解釈 |
| 3 | `(EPC - H) - B` | 内部状態 - 位相（生物学的） | 相転移・秩序変数の議論 |

重要なのは、

- **式は1つ**
- **読解が複数**
- **正準は1つ**

という順序を崩さないこと。

---

## 7. 実務的な結論

### 7.1 正準読解

`origin story` に従い、

```text
EPC - (H + B) = cost - gain
```

を canonical とする。  
ただしこの `B` は generic な構造利得ではなく、採用メモに従う限り **Betti side の位相項、実質 `β₁`** と読む。

### 7.2 二次読解

以下は secondary interpretation として保持する:

- `(EPC - B) - H`: Helmholtz-like / structure-entropy
- `(EPC - H) - B`: phase-transition-like / internal-state-topology

### 7.3 書き方の原則

- spec では混ぜない
- paper 本文では reading 1 を使う
- reading 2/3 は appendix / discussion / thinking note に置く

これにより、式の多義性を **強さ** として使いつつ、**曖昧さ** に落とさずに済む。

---

## 8. まとめ

geDIG 式の強さは、同じ式が

- 工学的には `cost - gain`
- 理論的には `structure - entropy`
- さらに位相的には `internal-state - topology`

として読める点にある。

ただし、出発点はあくまで `origin story` の

```text
コストに見合う更新だけを受け入れる
```

という設計原理である。  
したがって、**第一の読み方を正準とし、他は解釈として従える** のが最も強い。
