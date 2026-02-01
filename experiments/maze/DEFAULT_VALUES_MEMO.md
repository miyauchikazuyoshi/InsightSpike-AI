# デフォルト値メモ (2026-02-01 修正後)

## SP/IG/GEDが正しく動作するための必須設定

### 変更されたデフォルト値

| パラメータ | 変更前 | 変更後 | 理由 |
|-----------|--------|--------|------|
| `timeline_to_graph` | `False` | `True` | タイムラインエッジがグラフに追加されないとSP計算でsub_beforeが常に0エッジになる |
| `linkset_mode` | `False` | `True` | linksetベースのエントロピー計算（論文準拠）を有効にする |
| `theta_ag` | `0.0` | `-1.0` | linkset_mode時にg0が負になるためAG gateが常に発動してしまう問題を回避 |

### 推奨設定（計画書S0ベースライン準拠）

```bash
python experiments/maze/run_experiment_query.py \
  --maze-size 25 --max-steps 60 --seeds 3 \
  --lambda-weight 0.5 --max-hops 15 \
  --sp-scope union --sp-hop-expand 3 --sp-boundary trim \
  --eval-all-hops \
  --linkset-mode \
  --theta-ag -1.0 --theta-dg 0.0 \
  --top-link 1 --commit-budget 0 \
  --norm-base link \
  --output results/summary.json \
  --step-log results/steps.json
```

### 各パラメータの意味

#### SP計算関連
- `--sp-scope union`: before/afterの両グラフのノード合集合を使用
- `--sp-hop-expand 3`: SPサブグラフを追加で3hop拡張
- `--sp-boundary trim`: ターミナルエッジを削除

#### IG計算関連
- `--linkset-mode`: linksetベースのエントロピー計算（論文準拠）
- `--norm-base link`: 正規化分母としてS_linkを使用

#### マルチホップ評価関連
- `--eval-all-hops`: 全hopを評価（early stopなし）
- `--theta-ag -1.0`: AG gateを実質無効化
- `--max-hops 15`: 最大15hopまで評価

#### グラフ構築関連
- `--timeline-to-graph`: タイムラインエッジ（Q_prev→dir→Q_next）をグラフに追加（デフォルトON）
- `--no-timeline-to-graph`: タイムラインエッジをグラフに追加しない

### 修正されたバグ

1. **evaluator.py try-except-elseバグ** (line 882-903)
   - 問題: `else:`がif文ではなくtry-exceptに属していたため、sp_hが上書きされていた
   - 修正: インデントを修正し、if文のelseとして正しく動作するように

2. **SP計算でsub_beforeが0エッジ**
   - 問題: timeline_to_graph=Falseでポジション間エッジがなく、k-hopサブグラフが孤立
   - 修正: timeline_to_graphをデフォルトTrueに変更

3. **エントロピー計算が0**
   - 問題: linkset_mode=Falseでlinksetベースの計算がスキップされていた
   - 修正: linkset_modeをデフォルトTrueに変更

4. **マルチホップ評価がスキップ**
   - 問題: g0が負（IG貢献が大きい）の場合、theta_ag=0.0でAG gateが発動
   - 修正: theta_agをデフォルト-1.0に変更
