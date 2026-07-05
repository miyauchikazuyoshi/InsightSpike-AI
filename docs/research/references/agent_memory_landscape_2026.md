# リファレンス: LLM エージェント長期記憶(agent memory)地形図 2026

**種別**: 外部研究のリファレンス(deep-research 成果 + 精読者の読み方註釈)
**調査日**: 2026-07-05(スナップショット。§0 の時間感度注意を参照)
**用途**: geDIG を「記憶の自律管理」応用軸(agent memory)へ位置取りする際の一次参照。
**参照元**: [strategy_memory_insight_roadmap_20260705](../thinking/strategy_memory_insight_roadmap_20260705.md)
**手法**: deep-research(5 角度・23 ソース fetch・114 主張抽出・25 を 3 票敵対的検証・24 confirmed / 1 killed → 14 統合)。

> **この文書の読み方**: 各節は【事実】(deep-research が一次情報源で検証した引用・数値)と、
> 【読み方】(精読者=Claude の解釈・geDIG への含意・確定前の宿題)を分離してある。
> 【事実】は arXiv/公式ドキュメントに遡れる。【読み方】は判断であり、鵜呑みにしないこと。

---

## 0. 最初に読む留保(deep-research の caveats を人間可読に再構成)

1. **時間感度が極めて高い**。主要ソースの多くが直近 6 ヶ月内(BEAM=ICLR2026、MemoryArena=ICML2026、
   AgeMem=ACL2026、cross-trajectory abstraction 系=2025 後半–2026)。本地形図は 2026-07 の断面で、
   特に「insight 評価の空き地」はフロンティアとして認識され始めており**月単位で埋まりうる**。
2. **「空き地」= 網羅的不在証明ではない**。空き地の主要証拠(§6)は arXiv 2603.07670(単著・査読なし
   プレプリント)の**地図に関する不在**であって、分野全体の不在ではない。Zep/Graphiti の
   temporal KG のように「地図に載らない占有地」が現に存在する。§9 の逆引き検証が新規性主張の前に必須。
3. **商用実装(OpenAI/Anthropic/Google の 2025–2026 メモリ機能)は本調査の最大の残穴**。
   検証できたのは 2024 年末版 ChatGPT/Coze の劣化数値のみ(2025 の ChatGPT メモリ改修後ではない)。
4. **LoCoMo の規模数値は検証で反証(1-2)された** — abstract 記載と公開データセット実態の乖離疑い。
   規模を引用する際は公開データセットを直接確認すること。LoCoMo の答え合わせ鍵の 64% が誤りという
   第三者監査(penfieldlabs)も存在。
5. sleep-time compute の性能数値(〜5x、+13/18%)は著者自作ベンチの自己報告(Letta に商業的利害)。

---

## 1. ベンチマーク地形 — 世代交代が起きている

### 【事実】
- **LongMemEval**(ICLR 2025、arXiv 2410.10813、500 問):チャット記憶を 5 能力に分解 —
  情報抽出/マルチセッション推論/時間推論/知識更新/abstention。商用メモリの劣化を定量化:
  持続対話で 30% 低下、ChatGPT −37%(0.918→0.577)、Coze −64%(0.918→0.330)。
- **LoCoMo**(ACL 2024、arXiv 2402.17753):QA 5 タイプ(single/multi-hop/temporal/commonsense/
  adversarial)。multi-hop(≈14.6%)が「複数セッションからの情報統合」を要求 = LoCoMo 内で
  最も insight に近い要素。ヒトに 56% 及ばず(temporal で 73% 差)。**ただし規模数値は反証・監査あり(§0)**。
- **BEAM**(ICLR 2026、arXiv 2510.27246):100 会話(各 100K–10M トークン)・2,000 人手検証質問・
  10 能力(新規 3: 指示追従/事象順序/**矛盾解消**)。LoCoMo/LongMemEval を名指しで
  「短セッション連結は人工的でセグメント分離が容易 → 真の長距離推論を不要にする」と批判、
  **単一連続ナラティブへ転換**。
- **MemoryArena**(ICML 2026、arXiv 2602.16313):記憶評価を 4 エージェントドメインに埋め込み、
  後続サブタスクが先行学習に依存。LoCoMo ほぼ満点のモデルが 40–60% に崩落、厳密 Success Rate は
  平均 0.12–0.23(Letta 0.15、Mem0 0.14、Claude-Sonnet-4.5 0.19、GPT-5.1-mini 0.16)。

### 【読み方】
- **受動的 recall は飽和、能動的・意思決定関連の記憶使用は未解決** — この分離が 2026 の主戦線。
  geDIG の迷路実験(記憶を使って行動を選ぶ=能動的)は、偶然にも**飽和した側ではなく未解決の側**に
  最初から立っている。LoCoMo で SOTA を狙うのは飽和側への遅参 = §ロードマップの「早くRAGに行かない」判断を追認。
- **評価軸は「連結された別セッション」→「単一連続ナラティブ」へ**。geDIG の合成迷路(段階 2:
  warmup1=A 領域、warmup2=B 領域)は BEAM の批判(セグメント分離が容易)を受ける設計に見えるが、
  迷路の場合は**同一空間の別領域**なので分離不可能(A と B は物理的に地続き)。BEAM の批判は
  対話の人工的セッション境界への批判であり、迷路の領域封鎖には当たらない — この区別を論文で明示すべき。
- 参照時は BEAM の**矛盾解消**と MemoryArena の**厳密 SR**を「越えるべき実在のバー」として使う
  (v3 でタイプルータが実在バーになったのと同じ役割)。

---

## 2. 手法系譜 — MemGPT → LLM 駆動グラフ → 政策学習

### 【事実】
- **MemGPT**(arXiv 2310.08560 系):OS の仮想メモリの借用(コンテキスト=RAM、外部ストア=ディスク、
  ページング)。2603.07670 サーベイが「OS 設計者が数十年前に完成させた仮想メモリの借用」と位置づけ。
- **A-MEM**(NeurIPS 2025、arXiv 2502.12110):Zettelkasten 型。新記憶を構造化ノート(記述/キーワード/
  タグ)化 → cosine top-k 近傍を候補に **LLM プロンプトがリンク生成と既存ノートの進化を判断**。
  **リンク判断を統治する類似度閾値・グラフ指標・定量基準は一切なし**。
- **AgeMem**(ACL 2026、arXiv 2601.01885):store/retrieve/update/summarize/discard の 5 記憶操作を
  ツール化し、パイプライン全体を **RL(3 段階訓練、step-level GRPO)で最適化**。
- **記憶操作の 6 原子**(サーベイ 2505.00675 v3):Consolidation / Updating / Indexing / Forgetting /
  Retrieval / Condensation(v1-2 は Compression)。表現軸 = **parametric(重み内・編集系)vs contextual
  (外部明示)**。
- **制御方策の 3 軸**(サーベイ 2603.07670):ヒューリスティック / プロンプト自己制御(LLM 判断)/ 学習(RL)。

### 【読み方】
- **制御方策の 3 択に「解析的ゲージ(情報理論量)による統治」という第 4 の選択肢が空いている**
  — これが geDIG の座。A-MEM は「LLM 判断」、AgeMem は「RL 学習」。geDIG の F<0 ゲートは
  「学習も LLM 呼び出しもなく、閉じた式で accept/restructure/forget を決める」= 第 4 の軸。
  **ただし §9 の逆引き(MDL/Bayesian surprise で駆動する先行がないか)を確定してから主張すること**。
- geDIG の Sleep-RAG 構想(merge/split/prune を F でゲート)は、6 原子操作のうち
  Consolidation/Updating/Forgetting/Condensation を**単一のゲージで統一する**提案 =
  分野が別々の操作として並べているものを 1 本の式に畳む。これは新規性の主張点になりうる。
- parametric vs contextual 軸で geDIG は完全に **contextual(structured=知識グラフ)側**。
  基礎モデル軸(重み内学習の制御)へ行くなら parametric 側への越境が必要 = grokking_curl の位置。

---

## 3. sleep-time compute は geDIG の sleep とは別物(重要な区別)

### 【事実】
- **sleep-time compute**(arXiv 2504.13171、2025-04、Letta 創業者 Packer/Wooders + UC Berkeley
  Stoica/Gonzalez = MemGPT 系譜)の正準定義:「クエリ到着**前に**コンテキストについてオフラインで
  思考し、予想クエリの有用量を事前計算して test-time 計算を削減する」。
  **この論文の範囲では長期記憶ストアの再編・consolidation ではない**。
- 効果:自作ベンチ(Stateful GSM-Symbolic / AIME)で test-time 計算 〜5x 削減、+13/18% 精度。
  対話メモリベンチ(LoCoMo 等)では**未評価**。
- 著者が境界条件を明言:**効果はクエリ予測可能性と相関、予測不能な設定では標準 test-time scaling が優る**。

### 【読み方】
- **名前は衝突するが機能が違う**。sleep-time compute =「クエリを予期して答えを前計算」(予測的)。
  geDIG の sleep =「クエリ非依存に記憶構造を再編」(構造的)。著者自身が
  「クエリ予測可能性に依存」と限界を引いており、**クエリ非依存のオフライン記憶再編は
  この系譜の定義域の外側に開いている** — これは deep-research の明示的な発見。
- 論文で geDIG の sleep を書くときは、**必ず sleep-time compute との差分を 1 段落置く**
  (混同を招く名前なので)。「我々の sleep は Packer et al. 2025 の予測的事前計算ではなく、
  クエリ非依存の構造再編である」と明示。
- 迷路の Wake-Sleep-Wake は既にこの「クエリ非依存の構造再編」を実装している(v2 で実証済み)。
  つまり geDIG は分野が空けている定義域に、実装と証拠を既に持っている。

---

## 4. オープン課題マップ(サーベイ 2603.07670 の明示的未解決 = geDIG の狙撃対象)

### 【事実】研究質問の 5 課題すべてが一次ソースで未解決と明言:
1. **統合**:「hoarding(全部貯めてノイズに溺れる)と amnesia(圧縮しすぎて希少事実喪失)の間を振動」
2. **忘却**:「hard time-based expiration, storage-limit eviction, or nothing at all」= 粗雑な処理のみ
3. **自己確証ループ**:reflective memory の self-reinforcing error(誤った結論が呼び出し経路を永久回避し
   反証を集められない = **error entrenchment**)
4. **セッション横断**:cross-session coherence は「a distinct—and largely unsolved—challenge」
5. **検索**:semantic 類似 + 時間順序 + 因果グラフ走査 + 反実仮想関連性のハイブリッドは「largely unexplored」
- **矛盾解消**(BEAM):全手法で 10 能力中最弱(0.006–0.05)、著者自ら「challenging open problem」。

### 【読み方】geDIG の各機構がどの課題に刺さるか:
| 分野のオープン課題 | geDIG の対応機構 | 状態 |
|---|---|---|
| 統合(hoarding↔amnesia の振動) | F<0 ゲートで accept を判定(振動を式で律する) | 迷路で write-gate 実証済み(98%圧縮) |
| 忘却(粗雑な expiration のみ) | **F>0 が続くエッジを刈る(origin story 167 行)** | **未実装 = 動的迷路の主役、最大の狙い目** |
| 自己確証ループ(error entrenchment) | **ダークルーム対策(canary + held-out gold)** = 6月 Sleep-RAG メモに設計済み | 設計済み・未実装 |
| セッション横断 | 合成迷路(段階 2)/ Case A 検出 | 設計済み・未実装 |
| 矛盾解消(全手法最弱) | Case C(Δβ₁=−1、ループ消滅)= 古い構造の解体 | 理論のみ、未着手 |

- **「忘却」が最も鮮烈な空き地**。分野は「時間切れ削除か、容量上限追い出しか、何もしないか」しかない
  と自認している。geDIG の **F 裁定エビクション**(コストに見合わないエッジを刈る)は、この粗雑さへの
  直接の対案。そして origin story 167 行が半年前にこれを書いていた。**動的迷路実験の戦略的価値が
  この deep-research で客観的に裏付けられた**。
- error entrenchment は、あなたが 6 月に FEP ダークルーム問題として自力で予見していたものと**同一**。
  分野が 2026 のサーベイでようやくオープン課題に挙げたものを、設計メモに先に持っている。

---

## 5. geDIG の位置取り — 埋まっている場所と空き地

### 【事実(deep-research の結論)】
- **埋まっている**:グラフ記憶(A-MEM/Zep/Graphiti)、LLM 駆動リンク生成(A-MEM)、OS 型管理(MemGPT)、
  RL 政策学習(AgeMem)。
- **空き地(a)**:リンク生成・統合・忘却を**定量的・情報理論的ゲージで統治**する機構は、A-MEM にも
  2026 サーベイの地図にも不在(free energy / information gain / information-theoretic への言及を
  271KB 全文検索でゼロ確認)。
- **空き地(b)**:**独立断片の接続から新結論を導く insight の明示的評価**は全ベンチマークで
  「述べられた情報の再結合(aggregation/comparison)」止まり。進化系サーベイ(2605.06716)が
  cross-trajectory abstraction を「2025 後半にようやく coherent な方向として出現、評価は著しく不十分」と自認。

### 【読み方 — geDIG にとって何を意味するか】
- **2 つの空き地は、geDIG の 2 つの主張とちょうど一致する**。(a)=「when to accept/restructure/forget を
  ゲージで決める」(v6.1 Abstract の主張そのもの)、(b)=「コンテキスト間の閃き」(あなたが今週表明した
  最終目標)。分野の空き地を狙って作ったのではなく、**独立に育てた理論が分野の空き地に後から一致した**
  — これは position の強さの良い兆候。
- ただし **§9 の宿題を果たすまで「空き地」を論文で断言しない**。特に(a)は「情報理論の語彙での逆引き」
  が未実施。MDL でメモリを剪定する研究、Bayesian surprise で新規性を測る研究は**存在する可能性が高い**
  (これらは古い概念)。geDIG の新規性は「free energy 様ゲージが**無い**」ことではなく、
  「**β₁(位相)を含む 3 項ゲージで、迷路→RAG を横断して実装・検証した**」ことに置くのが安全。
- (b)は最も守りやすい空き地。「aggregation/comparison を超えて、独立断片から**新規結論**(既存の
  どの断片にも書かれていない命題)を生成し評価する」ベンチは実在しない。これは geDIG の Case A 検出
  (2 経路が閉じて Δβ₁=+1)を**評価指標として提案できる**ことを意味する — 手法だけでなく
  **ベンチマークの提案**が空いている(分野が評価を欲しているのに無い)。

---

## 6. 精読者の総合判断(署名付き)

1. **応用軸の名称は「agent memory」で確定**。RAG は飽和側、agent memory の忘却・矛盾・横断は未解決側。
   看板の掛け替えは正しく、しかも未解決側に自分の実装が既にある。
2. **最優先の狙撃対象は「忘却」**。分野が最も粗雑(§4)、geDIG が最も鋭い対案(F 裁定)、
   origin story が半年前に予言、動的迷路で安く検証できる。**次の迷路実験(動的迷路)の戦略的正当性が
   外部から裏付けられた**。
3. **insight 評価(空き地 b)は手法だけでなくベンチマーク提案の機会**。長期的には geDIG の
   「Case A 率」を cross-context insight の測定指標として世に問える。
4. **sleep-time compute との名前衝突に注意**(§3)。論文で必ず差分を明示。
5. **確定前の宿題(§9)を果たすまで新規性を過大主張しない**。誠実性文化の一貫。

---

## 7. 一次情報源リスト(検証済み、angle 別)

**ベンチマーク**: LongMemEval [2410.10813] / LoCoMo [2402.17753, snap-research.github.io/locomo] /
BEAM [2510.27246] / MemoryArena [2602.16313] / MemoryAgentBench [2507.05257]
**サーベイ**: 記憶6操作 [2505.00675] / 3軸タキソノミー・オープン課題 [2603.07670] /
Storage-Reflection-Experience [2605.06716] / 旧サーベイ [2404.13501]
**手法**: A-MEM [2502.12110] / AgeMem [2601.01885] / sleep-time compute [2504.13171, docs.letta.com/…/sleeptime]
**商用/その他**: Graphiti [github.com/getzep/graphiti] / ChatGPT memory [openai.com/index/chatgpt-memory-dreaming] /
Zep 批判 [blog.getzep.com/…] / LoCoMo 監査 [dev.to/penfieldlabs/…] / mem0 benchmarks [mem0.ai/blog/…]
**空き地プローブ(未精読、§9 で確認予定)**: EvolMem [2601.03543] / LongMemEval-V2 [2605.12493] /
[2406.14546] / [2606.01223] / [2605.06716]

---

## 8. reproducibility

deep-research 生成物(全 findings・evidence・vote): この session の workflow 出力
(`wf_444df44c-54b`)。stats: 5 角度・23 ソース・114 主張抽出・25 検証・24 confirmed / 1 killed
(LoCoMo 規模数値)・14 統合。105 エージェント呼び出し。

---

## 9. 確定前の宿題(openQuestions — 新規性主張の前に必須)

1. **情報理論ゲージでのリンク/忘却/統合駆動の逆引き探索**(最重要)。サーベイでの不在は不在証明でない。
   「MDL memory pruning」「Bayesian surprise novelty memory」「free energy memory consolidation」
   「information gain retrieval」等の語彙で逆引きし、**先行がないことを能動的に確認してから**
   geDIG の新規性を(a)に置く。無ければ(a)、あれば新規性を「β₁ 含む 3 項 × クロスドメイン実証」に移す。
2. **EvolMem(2601.03543)の second-order inference と LongMemEval-V2(2605.12493)の中身**を精読。
   空き地(b)が既に埋まりつつある可能性の一次確認。
3. **Zep/Graphiti の temporal KG がどこまで占有しているか**。geDIG の差別化を「ゲージ駆動の構造操作
   (統合・忘却・リンク判断)」に絞れるか、検索面でも重複するかの精査。
4. **OpenAI/Anthropic/Google の現行(2025–2026)メモリ機能の内部アプローチとベンチスコア**(残穴)。
