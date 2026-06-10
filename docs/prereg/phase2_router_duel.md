# 事前登録: Phase 2 ルータ対決（F-routing vs 競合シグナル）

> **日付**: 2026-06-10（実験実施前にコミット）
> **状態**: DRAFT — 予算・モデル選定の DECISION 2点を確定したら FROZEN に変更し、以後は追記のみ（本文の書き換え禁止）
> **前提**: Phase 1 オラクル分析（`docs/audits/oracle_routing_ceiling.md`）で gpt-4o +8.0pt → INVEST ゲート通過

## 1. 主張（検証対象）

> 安価なグラフ構造信号 F は、「この問題は軽い計算（Hybrid-E1）で正解できるか」を、信頼度・類似度ベースの既存信号より正確に予測する。

## 2. 理論の事前予測（F理論が正しければ成り立つこと）

- **P1**: F（`extended_f`）の routing AUROC > 0.5（チャンスを超える）
- **P2**: F の AUROC > 問題タイプルータ（comparison→Hybrid / bridge→IRCoT）の AUROC — F はタイプ情報を超える構造情報を持つ
- **P3**: F の AUROC ≥ logprob 信頼度 および 検索スコアマージン — 「構造信号」が「自信・類似度」に勝る
- **P4**: end-task Pareto（EM × LLM呼び出し数）で、F-routing が同一呼び出し予算のランダム振り分けを上回る

## 3. 反証条件（外れたら理論側の敗北として記録する）

| 条件 | 帰結 |
|------|------|
| P3 が両方に対して不成立（logprob にも margin にも AUROC で負け） | **「構造信号」主張を RAG 側で取り下げ**（レビュー§4の撤退条件） |
| P2 不成立（タイプルータに負け） | F の寄与は問題タイプの代理に過ぎないと記録し、主張を「タイプ＋αの安価な近似」に縮約 |
| P1 不成立 | ルーティング信号としての F を全面棄却 |

事後説明による矛盾吸収は行わない。予測が外れた場合は本ファイルに「敗北記録」を追記する。

## 4. 実験設計

### Stage A — 遡及分析（コスト $0、既存ログのみ）
- データ: `experiments/hotpotqa_v2/results/500q_hybrid_e1_{4o,mini}/results.jsonl`（`extended_f`, `question_type`, `em` 記録済み）
- 目的変数: 「Hybrid-E1 が正解したか」（em=1）
- 比較: F の AUROC vs タイプルータ vs （取得可能なら）検索スコアマージン
- 注意: logprob はログに無いため Stage A では比較不能 → P3 は Stage B で判定

### Stage B — 前向き実験
- ルータ: (a) LLM logprob/自己申告信頼度 (b) 検索スコアマージン・エントロピー (c) Adaptive-RAG型小型分類器 (d) 同一System-2発火率のランダム (e) 問題タイプルータ (f) **F-routing**
- データ: HotpotQA dev（**DECISION-1**: 既存500問で先行 → 全7,405問への拡大可否とモデル［gpt-4o-mini中心か］を予算で確定）
- 追加: MuSiQue（**DECISION-2**: mini系の中間判定解消用。v11設計＝50パラ拡大コンテキスト＋事前構築グラフ）
- 統計: paired bootstrap（AUROC差）、McNemar（EM差）。主要指標は **gpt-4o 系の routing AUROC**（これ1つを主検定とし、他は探索的と明記＝多重比較問題の回避）

## 5. 主要指標（変更禁止）

1. **Primary**: gpt-4o における F-routing の AUROC（目的変数: Hybrid正解可否）と、logprob・margin の AUROC との paired bootstrap 差
2. Secondary: EM × LLM呼び出し数の Pareto、s_PSZ、タイプ別内訳

## 6. 既知のリスク

- Hybrid-E1 の `extended_f` は System 選択後に記録された値であり、選択バイアスを含む可能性（Stage A の限界として明記; Stage B では振り分け前に全信号を記録する）
- オラクル +8.0pt は2腕選択の上限であり、実ルータの取り分はこれを大きく下回り得る
