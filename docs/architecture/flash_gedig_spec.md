# Flash-geDIG Specification

Flash-geDIGは、Transformer attentionをCPU/NetworkXへ移さずに評価する
torch-native APIである。すべての演算は微分可能だが、次の2種類の量を混同しては
ならない。

- **canonical delta F**: before/afterの変化量。リポジトリ共通規約はlower is better。
- **single-state structural profile**: 1つのattentionを測るFlash固有の絶対profile。
  普遍的な良否方向はなく、実験ごとに目的方向を明示する。

## Functional API

### Canonical before/after delta

```python
import torch
from insightspike.gedig import compute_delta_f_score

before = torch.softmax(torch.rand(1, 12, 64, 64), dim=-1)
after = torch.softmax(torch.rand(1, 12, 64, 64), dim=-1)
mask = torch.ones(1, 64)

result = compute_delta_f_score(
    before,
    after,
    mask=mask,
    lambda_param=1.0,
    gamma=0.5,
)

print(result.F_mean)       # scalar; lower is better
print(result.delta_epc)
print(result.delta_h)
print(result.delta_sp)
```

このAPIは`gedig.adapters.transformer.TransformerFEval`へ直接委譲する。
identical before/afterでは全delta項とFが0になる。

### Single-state structural profile

```python
from insightspike.gedig import compute_structural_profile

profile, metrics = compute_structural_profile(
    after,
    attention_mask=mask,
    temperature=0.1,
    percentile=0.9,
)

print(profile.mean())
print(metrics["epc"], metrics["h"], metrics["sp"])
```

`compute_f_score(attention, ...)`は旧利用者のために残す互換wrapperであり、
同じ数値を歴史的な`delta_epc` / `delta_h` / `delta_sp`キーで返す。
新規コードでは用途に応じて上の2 APIを選ぶ。

## Loss API

```python
from insightspike.gedig import FlashGeDIGLoss

# 旧挙動を明示: single-state profileを最大化
regularizer = FlashGeDIGLoss(
    alpha=0.01,
    objective="maximize",
)

outputs = model(inputs, output_attentions=True)
loss = outputs.loss + regularizer(
    outputs.attentions,
    attention_mask=inputs["attention_mask"],
)
loss.backward()
```

`FlashGeDIGLoss`はsingle-state profile用であり、返り値は次のとおり。

| `objective` | 返す値 | 用途 |
|---|---:|---|
| `"minimize"` | `+alpha * profile_mean` | profileを下げる実験 |
| `"maximize"` | `-alpha * profile_mean` | profileを上げる実験。後方互換の既定 |

canonical delta Fを最小化する場合は`compute_delta_f_score(...).F_mean`を
直接task lossへ加える。profile最大化という実験固有の構成は、canonicalな
「lower delta F is better」という判断規約を変更しない。

## Flash Profile Approximation

| 成分 | Flash profile近似 | 計算量 |
|---|---|---:|
| EPC-like density | sigmoid soft threshold後のedge密度 | $O(N^2)$ |
| Entropy | attention行列全体の正規化Shannon entropy | $O(N^2)$ |
| SP-like efficiency | 隣接行列の有限回matrix power | $O(kN^3)$ |
| Clustering | soft adjacencyの`trace(A^3)` | $O(N^3)$ |

profileとcanonical delta adapterは、percentile、temperature、entropy、SPの
定義が異なる。profileを2回計算して差し引く方法でdelta Fを作ってはならない。

## Compatibility

- `compute_f_score`の引数、tuple戻り値、歴史的metricキーは維持する。
- `FlashGeDIGLoss`の既存5位置引数は維持し、`alpha`と`objective`はkeyword-only。
- 数値固定、identical delta、adapter値/勾配、loss符号を回帰テストで検証する。
