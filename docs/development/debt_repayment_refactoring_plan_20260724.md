---
status: completed
created: 2026-07-24
updated: 2026-07-25
owner: repository-maintenance
---

# 技術的負債返済・段階的リファクタリング計画（2026-07-24）

**文書ステータス**: COMPLETED
**開始日**: 2026-07-24
**対象**: 公開 API、設定、永続化、geDIG API、`MainAgent`、実験 adapter 境界
**原則**: 科学的再現性と既存公開契約を保ったまま、小さな挙動固定テストを先に置いて進める

## 1. 目的

この計画は、以下の既知負債を段階的に返済する。

1. 公開 `create_agent()` が「ready-to-use」という契約に反して未初期化で返る。
2. Legacy CLI / 互換 `cycle()` が設定なしで `MainAgent()` を生成する。
3. DataStore 設定が `MainAgent` の実体依存へ結線されず、アプリの永続化契約も
   `save()` / `load()` と `save_state()` / `load_state()` の間で切れている。
4. 古い設定キーがネストモデルで黙って無視される。
5. geDIG の EPC・AG/DG・F の符号、および Flash API の損失方向に文書・実装差がある。
6. `make test` が正準 `src/gedig/tests/` を実行しない。
7. `MainAgent` がオーケストレーション、初期化、永続化、結果集約を一つの巨大クラスで担う。
8. 実験コードと正準 `src/gedig/` adapter の移行状態が部分的で、再現用コードとの境界が曖昧。

## 2. 保護対象と非交差ルール

開始時点で以下は利用者の未コミット研究作業である。今回の負債返済では変更しない。

- `experiments/maze/graph_persistent_dg/sleep_propagate.py`
- `experiments/maze/qhlib/cli.py`
- `experiments/maze/qhlib/models.py`
- `experiments/maze/run_experiment_query.py`
- `docs/research/thinking/gedig_flow_field_edge_preference_20260704.md`
- `experiments/maze/run_v7_dgwire_smoke.sh`

追加ルール:

- v7 の科学的操作と、挙動不変を目標とするリファクタリングを同じ変更単位に混ぜない。
- `docs/prereg/` の凍結済み文書と再現用 experiment-local 実装は書き換えない。
- 旧 import path と公開シンボルは shim を残して維持する。
- F の確立済み規約は `docs/CANONICAL.md` を優先する。
- 既存の失敗・予備結果を、リファクタリングによって成功扱いへ変更しない。

## 3. ステータス定義

| 状態 | 意味 |
|---|---|
| `PLANNED` | 未着手 |
| `IN PROGRESS` | 実装または検証中 |
| `COMPLETED` | 受入条件を満たし、計画書と検証結果を更新済み |
| `BLOCKED` | 外部判断または新しい権限が必要 |

## 4. 実施フェーズ

| ID | フェーズ | 状態 | 主な受入条件 |
|---|---|---|---|
| P0 | ベースライン固定と characterization test | `COMPLETED` | 既知の壊れた契約を再現するテストを追加し、正準71件と軽量smokeの基準を維持 |
| P1 | 公開API・初期化・旧エントリポイント修復 | `COMPLETED` | `create_agent()`直後に知識投入可能。Legacy CLI / `cycle()` が有効な設定を使用 |
| P2 | DataStore注入・Wrapper永続化修復 | `COMPLETED` | filesystem/memoryの生成と注入、learn→save→reloadの契約を統合テストで確認 |
| P3 | 設定移行と未知キー診断 | `COMPLETED` | 旧キーを互換変換し警告。未知ネストキーを黙って無視しない。Pydantic v1/v2互換 |
| P4 | geDIG意味論・Flash API・テスト入口統一 | `COMPLETED` | F目的方向を明示、EPC/AG/DG表記を正準化、通常コマンドでcore 76件も実行 |
| P5 | `MainAgent`責務抽出 | `COMPLETED` | 公開class/importを維持したまま、初期化・永続化・結果集約・設定参照を協調サービスへ抽出 |
| P6 | adapter移行境界・全体検証 | `COMPLETED` | 現行実験と凍結再現コードの境界を文書化し、独立oracle・golden trace・対象回帰テストを通過 |

## 5. フェーズ別の変更計画

### P0: ベースライン固定

変更前の挙動を characterisation test で固定する。バグ修正対象は、まず期待する契約を
テストとして書き、そのテストが修正前に失敗することを確認してから実装する。

対象:

- `tests/unit/test_quick_start_overrides.py`
- `tests/unit/test_public_api_extras.py`
- 新規の公開runtime / wrapper / config互換テスト
- `src/gedig/tests/`
- `scripts/codex_smoke.sh`

受入条件:

- 正準core: `71 passed`
- cloud-safe smoke: `11 passed`以上（追加分は増加可）
- 利用者のv7差分に変更がない

### P1: 公開APIと初期化

方針:

- `create_agent()` は生成後に `initialize()` を呼び、成功しなければ明確に失敗させる。
- 初期化状態は `initialized` property に統一し、`_initialized` は内部互換として残す。
- `MainAgent` の設定なし生成を許すのではなく、旧呼び出し側で `load_config()` した設定を渡す。
- `process_question()` の遅延初期化は後方互換のため残す。

受入条件:

- `create_agent(provider="mock")` 直後の `add_knowledge()` が成功する。
- Legacy CLI の各生成箇所と互換 `cycle()` が設定エラーにならない。
- 既存のpublic import pathが変わらない。

### P2: DataStoreとアプリ永続化

方針:

- DataStore種別の正準名を定め、`memory` / `in_memory` を互換aliasとして正規化する。
- composition helperを設け、configからDataStore実体を一箇所で生成する。
- `create_agent()` から `MainAgent(config, datastore=...)` へ注入する。
- `InsightAppWrapper.save/load` を `save_state/load_state` へ接続する。
- Wrapperの`data_dir`指定が実際の保存先へ反映されることを確認する。
- 既存の追記操作と完全snapshot置換を分離し、`replace_episodes()`を追加する。
- episodeの`text/vec/c/metadata`と拡張属性を共通codecで往復させ、再読込後に
  L2 vector indexを再構築する。
- SQLiteはDBを正本、vector indexを再生成可能なcacheとして扱い、schemaは変更しない。

受入条件:

- memory / filesystemの双方がpublic APIとcomposition rootから生成可能。
- 一時ディレクトリで、知識投入→保存→新規Wrapper/Agent→再読込を確認。
- SQLiteの既存契約・スキーマは変更しない。
- 2件のsnapshotを2回保存しても、再読込後は重複せず正確に2件となる。
- `c`、metadata、vector、episode拡張属性が往復し、SQLite再起動後も検索可能。

### P3: 設定移行と診断

方針:

- Pydantic v1/v2で同じ挙動になる互換基底とdump/validate helperを導入し、
  v1で`model_config`が実フィールド化する問題を除去する。
- YAML読込後、Pydantic検証前に旧キーを正準キーへ変換する。
- 変換したキーごとに一度だけ警告を出す。
- 旧キーと新キーが同時にある場合は新キーを優先し、衝突を警告する。
- ネストモデルも未知キーを検出できるようにする。ただし既知の旧キーは移行後に検証する。
- `ConfigNormalizer`の選択的再構築を廃止し、正準セクションを欠落なく保持する。
- YAML保存はPathやv1 artifactを含まないsafe-load可能な表現へ正規化する。

初期移行表:

| 旧キー | 正準キー |
|---|---|
| `output.response_style` | 廃止。警告して無視 |
| `output.show_reasoning` | `output.include_reasoning` |
| `output.show_metadata` | `output.include_metadata` |
| `monitoring.enable_monitoring` | `monitoring.enabled` |
| `monitoring.track_memory_usage` | `monitoring.performance_tracking` |
| `monitoring.metrics_interval` | 廃止。警告して無視 |
| `logging.format` | 廃止。警告して無視 |
| `logging.file_enabled` | `logging.log_to_console` とは意味が異なるため廃止警告 |
| `datastore.type: in_memory` | `memory` factory aliasへ正規化 |
| `reasoning.episode_*_threshold` | `graph.episode_*_threshold` |

完了時の追加対応:

- source単位でmigrationしてから`preset < file < env < override`をmergeし、
  診断へsourceを記録した。
- `ConfigLoader`再利用時の保存先リークと、DataStore既定pathの明示/暗黙を
  save→reloadで取り違える問題を修復した。
- 数字だけのmodel名/pathを文字列のまま扱い、整数fieldの小数文字列を
  v1/v2で同じく拒否する型別env parserへ変更した。
- `ConfigNormalizer`とlegacy adapterを全設定セクション保持の一括検証へ統一した。
- 配布中の`config_examples/`全7 YAMLとREADMEをstrict canonical APIへ更新した。
- Layer3 message-passingの誤importを修復し、設定済みのpercentile/cacheを実装へ
  結線した。`max_hops`が常に1-hopで止まる探索バグも回帰テスト付きで修復した。
- 独立したindex設定を含む全設定モデルで未知キーを拒否するよう統一した。

受入条件:

- 現在の`config.yaml`を読み込み、移行された値と警告をテスト。
- 未知キーのタイプミスが例外または構造化警告として観測可能。
- preset / file / env / overrideの優先順位を維持。
- save→safe-loadの往復、およびPydantic v1/v2の双方を検証。

### P4: geDIG意味論とテスト入口

方針:

- 正準の判断方向は「lower F is better」のまま維持する。
- 現行Flash処理はbefore/after差分ではなく単一attentionの絶対profileであるため、
  既存挙動を`compute_structural_profile`として命名し、互換wrapperを残す。
- canonical delta APIは`compute_delta_f_score(before, after, mask)`として別に追加し、
  `TransformerFEval`へ委譲する。
- Flash正則化は実験固有のprofile最大化を許すが、`objective`で明示する。
- `FlashGeDIGLoss`へ係数を持たせる場合は`alpha`を実装し、既定値で後方互換を保つ。
- architecture文書のEPC誤記とAG/DG別名を正す。
- AG/DGの二段gateと、RAG/Transformerのedge partitionを別型・別名称へ分離する。
- `make test`で`tests/`と`src/gedig/tests/`をともに実行する。

完了時の追加対応:

- 旧単一状態値を`compute_structural_profile`へ分離し、旧
  `compute_f_score`の数値・tuple・metric keyを互換wrapperで固定した。
- canonical before/after APIは`TransformerFEval`の結果型をそのまま返し、
  非既定parameter、mask、値、勾配まで直接adapter呼出しと比較した。
- `FlashGeDIGLoss`へkeyword-onlyの`alpha`と`objective`を追加し、既定の
  profile最大化挙動と既存5位置引数を維持した。
- AG/DG event gateを`TwoStageGateDecision`、edge score分割を
  `EdgePartitionResult`へ分け、旧分類名はmutable thresholdを保つshimとした。
- 二閾値のdeadband edgeを`middle_score_edges`へ保持し、分割結果が入力edgeを
  欠落させない契約を追加した。
- EPCをEdit Path Cost、無修飾Fの第三項を確立済みSP、β₁を明示的な研究modeとして
  README・architecture・API文書で統一した。
- bare `pytest`、`make test`、coverage入口が通常testsとcore testsの双方を収集する
  ようにし、test packageのshadowとoptional依存skip判定を修復した。

受入条件:

- Flash lossのminimize/maximize両方向に勾配・符号テストがある。
- identical before/afterに対するdelta APIが0で、adapterと数値・勾配が一致する。
- 旧Flash profileの固定出力が変わらない。
- README Quick StartとAPI docが一致。
- 正準core 76件を通常のテスト入口から実行できる。

### P5: `MainAgent`責務抽出

公開 `MainAgent` はfacadeとして維持し、次の順で純粋な責務から抽出する。

1. 永続化
2. cycle結果の選択・集約
3. 初期化状態とcomponent lifecycle
4. 設定値の参照

候補サービス:

- `AgentLifecycle`
- `AgentPersistence`
- `CycleResultAggregator`
- `AgentConfigAccess`

このフェーズではL1〜L4のアルゴリズム自体は変更しない。

受入条件:

- `MainAgent`, `CycleResult`, `cycle`のimport pathと戻り型を維持。
- 既存テストpatch点を維持。
- 抽出前後で公開runtime、主要integration、core equivalenceが通る。

完了内容:

- 公開`MainAgent`をfacadeおよびL1〜L4 component ownerとして維持し、
  永続化、cycle結果集約、runtime lifecycle、設定参照を4サービスへ抽出した。
- `MainAgent.__new__()`を使うconstructor-free経路、同名の永続化facade、
  module-level provider差し替え、公開component属性の差し替えを維持した。
- LLM providerは解決直後に公開属性へ束縛し、後続初期化の失敗後も同一instanceを
  再利用する。repeat初期化時のprovider/memory/monitor順序と既存状態契約も固定した。
- LLM生成失敗時のfallbackを`AgentLifecycle`の単一実装へ統一した。
- cycleは最高`reasoning_quality`を選び、tieは先頭を維持する。学習hook完了後に
  `CycleResult`をmaterializeする旧順序、metadata、`query_id`契約を維持した。
- 設定値はlive sourceから再構成し、`_normalized_config`差し替えは変更fieldだけを
  overlayする。これによりgeDIG mode overrideとadaptive threshold追従を両立した。
- DataStore exact snapshot、empty load、index rebuild、graph戻り値、legacy fallbackの
  保存契約をcharacterization testsで固定した。
- `mainagent_behavior.md`、詳細API、directory structureを現在のfacade/service構成と
  `save_state()` / `load_state()`の`bool`契約へ同期した。

### P6: adapter境界と最終検証

方針:

- `src/gedig/`を新規コードの唯一の正準実装として維持。
- 凍結済みexperiment-localコードは削除せず、移行状態を正確に記録する。
- Transformer / RAG / Mazeごとに「委譲済み」「flag切替」「旧実装固定」を明記する。
- Transformer equivalenceは同一adapter同士の自己比較を廃止し、凍結legacy oracleと比較する。
- E2Eが未実行なら、等価性を超えて完了したとは記載しない。
- Mazeのactive evaluator移行は進行中v7と交差するため、この計画ではgolden traceと
  正確な移行状態の文書化までとし、切替自体はv7完了後の明示フェーズへ送る。

完了内容:

- Transformerの比較元を、現行adapterを再生成していた偽の比較から、commit
  `eed1ae4`由来の凍結legacy oracleへ変更した。比較元がunified adapterを生成して
  いないことをテストで明示し、SP / Betti、mask、非既定parameter、全delta、
  勾配、速度を独立実装間で確認した。T4の実験E2Eは再実行していないため、
  移行完了の根拠には含めない。
- RAGのactive AGHT経路が常に`RAGFEval`へ委譲していること、残存
  `use_unified` / `--aght-use-unified`が切替を行わないdeprecated no-opであることを
  code・CLI help・移行台帳へ明記した。R1〜R3は独立した手計算比較として維持し、
  歴史的なR4〜R6を今回の再検証済み結果としては扱わない。
- Mazeのactive evaluatorはexperiment-local legacy `GeDIGCore`固定のままとした。
  現行SP経路のg0 / gmin / best hop / selected edge / delta成分 / hop系列を
  deterministic golden traceで凍結した。正準`MazeFEval`との等価性や切替完了は
  主張せず、v7完了後の別フェーズへ送る。
- 最終監査で判明したmessage-passing集約契約も修復した。`mean`を正式値、
  既存`attention`を`FutureWarning`付きのmean互換aliasとし、その他の誤記は
  `ValueError`にした。定義のない真のattention実装は既存結果を変えるため、
  科学的仕様を決める別featureへ分離する。
- 最終横断検証では、2026-02-01のbaselineで既知だった`MainAgent`統合失敗
  11件を再監査した。今回のP2/P3で3件が既に解消され、残る8件も次の公開契約
  修復により`22 passed`へ移行した。
  - `add_document()`は文字列に加えlegacy document mappingを正規化する。
  - `learn()`の`episodes_added`は有効なepisode idから算出する。
  - 現行L3のno-opを更新済みと誤報せず、明示的な`True`だけを
    `graph_updated=True`として扱う。
  - Insight取得・検索はagent所有のregistryと現行`InsightFact` APIを使う。
  - 統合テストはLite-modeでL3を誤って期待せず、遅延L4 factory契約を検証する。
- `MIGRATION_PROGRESS.md`を現行の事実台帳へ更新し、Transformer / RAG / Mazeの
  「委譲済み」「no-op flag」「legacy固定」と、未実行E2Eを区別した。

最終受入条件:

- core 80件
- cloud-safe smoke
- 変更対象のunit/integration tests
- import / config / public API smoke
- `git diff --check`
- 利用者のv7差分が開始時点から意図せず変更されていない

## 6. 検証記録

| 日時 | フェーズ | コマンド/確認 | 結果 |
|---|---|---|---|
| 2026-07-24 | 開始前 | `PYTHONPATH=src .venv/bin/python -m pytest -q src/gedig/tests` | 71 passed |
| 2026-07-24 | 開始前 | `bash scripts/codex_smoke.sh`（`.venv/bin`優先） | 11 passed |
| 2026-07-24 | 開始前 | `git status --short` | v7関連4変更+2未追跡を保護対象として記録 |
| 2026-07-24 | P0 | `pytest -q tests/unit/test_debt_repayment_contracts.py` | 4 strict xfailed（P1×2、P2×2の既知負債を再現） |
| 2026-07-24 | P1 | 公開runtime対象11 tests | 9 passed、P2の2件のみstrict xfailed |
| 2026-07-24 | P1 | full-mode top-level import smoke | `insightspike.MainAgent`が実装classと一致、`create_agent` callable |
| 2026-07-24 | P1 | lite public runtime smoke | `create_agent()`直後にinitialized、`add_knowledge()`成功 |
| 2026-07-24 | P1 | `rg 'MainAgent\\(\\)' src` / `git diff --check` | 設定なし生成0件、問題なし |
| 2026-07-24 | P2 | DataStore/Wrapper/CLI/MainAgent対象tests | 21 passed |
| 2026-07-24 | P2 | public API・config pattern・実knowledge E2E対象tests | 8 passed |
| 2026-07-24 | P2 | changed Python `py_compile` / `git diff --check` | 問題なし |
| 2026-07-24 | P3 | Pydantic v1 設定・message-passing対象tests | 81 passed |
| 2026-07-24 | P3 | Pydantic v2 設定・message-passing対象tests | 65 passed、8 skipped（optional PyG未導入） |
| 2026-07-24 | P3 | `config_examples/`全7 YAMLを両Pydantic版でstrict load | 全件成功、migration diagnostics 0 |
| 2026-07-24 | P3 | changed Python `py_compile` / `git diff --check` | 問題なし |
| 2026-07-24 | P4 | `PYTHONPATH=src .venv/bin/python -m pytest -q src/gedig/tests` | 76 passed |
| 2026-07-24 | P4 | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/test_flash_gedig_api.py` | 10 passed |
| 2026-07-24 | P4 | torch import-block simulation | 56 passed、20 skipped（light CI想定） |
| 2026-07-24 | P4 | bare `pytest --collect-only -q` | 1046 collected、33 skipped。通常testsとcore testsを収集 |
| 2026-07-24 | P4 | `make -n test coverage` | 両入口が`tests/`と`src/gedig/tests/`を対象化 |
| 2026-07-24 | P4 | changed Python `compileall` / public import・type-hint smoke / `git diff --check` | 問題なし |
| 2026-07-24 | P5 | service contracts + geDIG pipeline | 29 passed（service 23、pipeline 6） |
| 2026-07-24 | P5 | public runtime / exact snapshot / query storage横断 | 17 passed |
| 2026-07-24 | P5 | service contracts + canonical core | 99 passed（service 23、core 76） |
| 2026-07-24 | P5 | agents `compileall` / full-mode public import / `git diff --check` | 問題なし |
| 2026-07-24 | P6 | `PYTHONPATH=src .venv/bin/python -m pytest -q src/gedig/tests` | 80 passed |
| 2026-07-24 | P6 | torch import-block simulation | 57 passed、23 skipped（light CI想定） |
| 2026-07-24 | P6 | Transformer独立oracle + Maze active trace | 15 passed |
| 2026-07-24 | P6 | message-passing通常版・最適化版・集約契約 | 20 passed（PyG導入環境） |
| 2026-07-24 | P6 | `PATH="$PWD/.venv311/bin:$PATH" bash scripts/codex_smoke.sh` | 16 passed |
| 2026-07-24 | P6 | bare `pytest --collect-only -q` | 1078 collected、33 skipped。通常testsとcore testsを収集 |
| 2026-07-24 | P6 | `compileall src tests` / `git diff --check` | 問題なし |
| 2026-07-24 | P6 | 保護対象確認 | v7関連4変更+2未追跡を保護したまま、今回のpatch対象外を維持 |
| 2026-07-24 | P6 | 未変更HEADの`test_main_agent_integration.py` | 11 failed、11 passed（既知baselineを独立展開して再現） |
| 2026-07-24 | P6 | 現行`test_main_agent_integration.py`（PyG / torchless） | 両環境とも22 passed |
| 2026-07-24 | P6 | MainAgent integration + 抽出service contracts | 45 passed |
| 2026-07-24 | P6 | 公開API・設定・永続化・Flash・message-passing・MainAgent横断 | 143 passed |
| 2026-07-24 | P6 | final `compileall src tests` / public import・config smoke / `git diff --check` | 問題なし |
| 2026-07-25 | コミット前 | `PATH="$PWD/.venv311/bin:$PATH" bash scripts/codex_smoke.sh` | 16 passed |
| 2026-07-25 | コミット前 | README契約・検証値の同期 / `git diff --cached --check` | 問題なし |

## 7. 変更履歴

| 日付 | 更新 |
|---|---|
| 2026-07-24 | 初版作成。P0を`IN PROGRESS`として開始 |
| 2026-07-24 | P0完了。実行可能な負債契約4件をstrict xfailで固定し、P1を開始 |
| 2026-07-24 | P1完了。公開初期化、top-level lazy export、Legacy CLI/cycle、optional PyG import境界を修復。P2を開始 |
| 2026-07-24 | 並行監査結果を反映。P3へPydantic v1/v2・normalizer・safe round-trip、P4へFlash profile/delta分離、P6へ真のlegacy oracle比較を追加 |
| 2026-07-24 | P2完了。DataStore生成を共通化し、quick start/CLI/Wrapperへ実体を注入。追記とsnapshot置換を分離し、episode codec、索引再構築、SQLite DB正本化を実施。P3を開始 |
| 2026-07-24 | P3完了。設定source別migration、strict nested schema、v1/v2互換、安全な往復保存、canonical examples、message-passing設定結線を完了。P4を開始 |
| 2026-07-24 | P4完了。Flash profile/canonical deltaを分離し、損失方向、gate/partition型、EPC/SP/β₁表記、標準テスト入口を統一。P5を開始 |
| 2026-07-24 | P5完了。`MainAgent`の公開facade/component ownershipを維持し、永続化・結果集約・lifecycle・live設定参照を4サービスへ抽出。P6を開始 |
| 2026-07-24 | P6完了。Transformerの真の凍結oracle比較、Maze active golden trace、RAG no-op flagの明文化、message-passing集約契約、移行台帳と全体検証を完了。本計画を`COMPLETED`へ更新 |
| 2026-07-24 | 最終横断検証で既知baselineの`MainAgent`統合失敗11件中8件が残存していることを再確認。公開compatibility debtを追加返済するためP6を再開 |
| 2026-07-24 | P6再完了。legacy document入力、learn結果、L3 no-op報告、Insight registry、Lite/L4統合テスト契約を修復し、既知MainAgent統合失敗を22件全通過へ改善。本計画を最終`COMPLETED`へ更新 |
| 2026-07-25 | README / README_ENを最終契約と検証結果へ同期。コミット対象110ファイルを確定し、利用者のv7作業中6ファイルは除外したまま最終化 |
