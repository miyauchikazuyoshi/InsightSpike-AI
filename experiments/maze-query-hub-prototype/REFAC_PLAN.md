# Maze Query-Hub: SP計算リファクタリング計画（DataStore/DistanceCache）

目的
- 大規模迷路（例: 100×100）まで見据え、SP（最短路相対短縮）評価を高速化。
- 既存実験（挙動/ログ/HTML）の互換性を維持しつつ、段階的に導入。

背景と課題
- 現状のマルチホップ評価は、各hopで候補エッジを1本ずつ仮追加し、その都度「before/afterサブグラフの平均最短路」を計算している。
- all_pairs_shortest_path_length 反復がボトルネック（候補×hop×サブグラフ規模）。

基本方針
- fixed_before_pairs（before固定ペア・相対短縮）を厳密に維持。
- before側の距離情報をキャッシュ再利用するDistanceCacheを導入：
  - 固定ペア集合（サンプル）と各ペア距離 d_before(a,b) を署名付きで保存。
  - 候補 e=(u,v) ごとの after 推定は、SSSP（BFS）距離 dist(a,u), dist(v,b) を合成して La を計算。
  - ΔSP = max(0, (Lb−La)/Lb)。
- DataStore（memory/SQLite）をバックエンドにし、ステップ間/試行間の再利用も可能に。

設計（要点）
- サブグラフ署名 sig:
  - anchors, hop（評価半径）、scope/trim設定、beforeサブグラフ（ノード/エッジ集合）のハッシュ。
- DistanceCache API（案）：
  - get_fixed_pairs(sig, before_subgraph) -> {pairs, Lb_avg}
  - get_sssp(sig, src, before_subgraph) -> dist_map（未保存ならBFS実行し保存）
  - estimate_sp_gain(sig, pairs, sssp_u, sssp_v) -> ΔSP
- Ecand上限: Top‑K（例:20〜50）で端点数/SSSP回数を抑制。
- Core準拠（union/trim/fixed_before_pairs）を踏襲。IGはlinksetモードを維持。

互換性
- 既定 off（--sp-cache を指定した時のみDistanceCache経由）。
- 既存のJSON出力・HTMLは維持。新規フィールド（例: sp_cache_stats）は後方互換の範囲で追加。
- Coreの _compute_sp_gain_norm 経路（sampling付）にフォールバック可能。

CLI/設定（追加）
- --sp-cache（既定: off）
- --sp-cache-backend {memory, sqlite}（既定: memory）
- --sp-cache-mode {core, cached}（既定: core）
  - core: CoreのSP計算に一本化（まずはここから）
  - cached: SSSP合成でΔSP推定（キャッシュ活用）
- --sp-cand-topk <int>（Ecand上限; 例: 32）
- --sp-pair-samples <int>（固定ペアサンプル数; 例: 400）

段階導入（フェーズ）
- フェーズA（最小統合）
  1) DistanceCache骨組み（memoryバックエンド）。
  2) HopGreedyEvaluatorからSP計算をDistanceCacheに委譲（sp-cache-mode=core）。
  3) 25×25 / H=15 / 1seed スモーク。Core直呼びと一致確認（ΔSP, g(h)）。
- フェーズB（キャッシュ適用）
  4) sp-cache-mode=cached 実装（署名生成→固定ペア抽出→SSSPキャッシュ→ΔSP合成）。
  5) Ecand Top‑K適用、pairサンプル・scope/hopを可調整。
  6) 性能計測: 25×25 → 100×100（短ステップ）→ 100×100（本番に近い設定）。
- フェーズC（永続化/運用）
  7) SQLiteバックエンド、TTL/LRU/GC導入。キャッシュヒット率/BFS回数/処理時間をログに集計。
  8) HTMLへのSP診断（Lb/La/ペア数/ヒット率）表示オプション。

テスト計画
- 単体テスト（小規模グラフ）
  - ΔSP一致: Core基準（fixed_before_pairs+union/trim）とDistanceCache（cached）が一致（許容誤差内）。
  - 署名安定性: anchors/hop/scope/trim/グラフが同一なら同じsig。
  - キャッシュ動作: SSSPキャッシュのヒット/ミス、LRU/TTLの動作（後日）。
- 統合テスト（迷路）
  - 15×15, 25×25で sp-cache=off/on(core)/on(cached) の3条件比較。
  - g0/gmin/ΔSPの統計が一致/近似すること（core vs cached）。
  - 速度比較（全体時間、1step平均、BFS回数）。
- 大規模ベンチ
  - 100×100, H=15, seeds=1〜3, step短縮で性能曲線確認。

ロールアウト方針
- まず --sp-cache-mode=core で機能経路を固める（挙動互換）。
- 問題なければ cached を既定化する案を検討（十分な性能/安定性が確認できた後）。

実装進捗メモ（更新していきます）
- [A-1] 2025-10-26: 本計画書を作成。
- [A-2] 2025-10-26: DistanceCache骨組み追加（memory）。HopGreedyEvaluatorにsp-cacheオプションとmode追加。core経路に委譲実装。
- [B-4] 2025-10-26: cachedモードのΔSP推定（SSSP合成）を実装（小規模スモーク整合）。
- [B-5] 2025-10-26: Ecand Top‑K制限のオプション（--sp-cand-topk）を実装。性能計測はこれから。
- [C-7] T.B.D.: SQLiteバックエンド、GC/統計集計。
