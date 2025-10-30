# Config Normalization & Legacy Phase4 Overview (2025-09)

## Purpose

Phase4 の目的は MainAgent 内に散在していた `isinstance(config, InsightSpikeConfig)` / `dict` / legacy 風オブジェクト分岐を 1 箇所へ集約し、**新旧設定のスムーズな共存**と将来の完全移行を容易にすることです。

## Key Components

| Component | File | Role |
|-----------|------|------|
| `compat_config.detect_config_type` | `src/insightspike/legacy/compat_config.py` | pydantic / dict / legacy 判定集中化 |
| `compat_config.normalize` | 同上 | dict を最小限フィールドで `InsightSpikeConfig` へ変換 (best-effort) |
| `NormalizedConfig` | `src/insightspike/config/normalized.py` | ミニマル読み出し用不変 view (embedding_dim など統一アクセス) |
| `MainAgent._compute_gedig` | `main_agent.py` | geDIG モード (full / pure / ab) ファサード |
| `GeDIGABLogger` | `src/insightspike/algorithms/gedig_ab_logger.py` | A/B 相関ログ + 閾値監視 + CSV エクスポート |

## Flow

```mermaid
flowchart LR
    A[User Config (dict / pydantic / legacy)] --> B(detect_config_type)
    B --> C{pydantic?}
    C -- yes --> NC[NormalizedConfig.from_any]
    C -- no --> D[normalize(dict→InsightSpikeConfig)] --> NC
    NC --> Agent[MainAgent]
```

## geDIG Modes

| Mode | 説明 | L3 Graph 必要 | 返却追加情報 |
|------|------|---------------|--------------|
| `full` | 既存フル計算 (L3 利用) | 任意 (無い場合 0 埋め) | なし |
| `pure` | 軽量 Pure 実装 (外部依存最小) | 不要 | なし |
| `ab` | `pure` + `full` 両方計算し相関追跡 | 片方のみでも継続 | `{"pure": {...}, "full": {...}}` |

## A/B Correlation Monitoring

- ローリング窓 (`window=100`)
- `pearson(gedig_pure, gedig_full)` を算出
- `min_pairs=30` 以上 & 相関 `< threshold(0.85)` で WARN ログ 1 回/カウント
- `flush_every` に到達で自動 CSV 書き出し (ヘッダ含め常に >=1 行)

### Basic Usage

```python
from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.config.models import InsightSpikeConfig

cfg = InsightSpikeConfig(llm={"provider": "mock", "max_tokens": 64})
agent = MainAgent(cfg)
agent.initialize()
# 別途 normalized config の gedig_mode を書き換え (テスト例と同様)
agent._normalized_config = agent._normalized_config.__class__.from_any(cfg, override={"gedig_mode": "ab"})

res = agent.process_question("Test correlation?", max_cycles=1)
agent.export_gedig_ab_csv("gedig_ab.csv")  # header-only でも存在保証
```

## Test Entry Points

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_legacy_config_detection.py -q` | config 判定ユニットテスト |
| `pytest tests/integration/test_gedig_pipeline.py::test_gedig_modes_smoke -vv` | geDIG 3 モード smoke |
| `pytest tests/integration/test_gedig_pipeline.py::test_ab_correlation_accumulates -vv` | A/B 相関蓄積 |

## Remaining Phase4 Tasks (Planned)

- [ ] `MainAgent` 受領時に dict 入力なら `normalize()` 自動適用 (手動差分を更に縮退)
- [ ] `GeDIGABLogger` の auto-flush 動作を検証する最小ユニットテスト追加
- [ ] 相関低下 (threshold 未達) を metrics エンドポイント or datastore にも記録
- [ ] README 英語/日本語表の統一レイアウト (geDIG モード部分を Features に昇格)

## Rationale

1. 可観測性: A/B 相関を早期に監視し pure 実装の逸脱を検知。
2. 移行容易性: `normalize()` による dict→pydantic 収束で後方互換コードの漸減。
3. テスト可能性: 統一インターフェイスにより smoke テストの簡素化 & 失敗箇所特定が容易。

## FAQ

**Q: CSV が空 (ヘッダのみ) だが失敗なのか?**  
A: 設計通り。A/B まだ十分なクエリが無い状態。`written >= 1` (ヘッダ) を成功条件としています。

**Q: Graph Reasoner 不在で `full` 0 値だが相関は?**  
A: その場合 `full` が定数に近く相関は NaN/低値になり得るが WARN 条件 (min_pairs>=30) 未満のうちは無視されます。

---
(Last update: 2025-09 Phase4 scaffold)
