# InsightSpike-AI Roadmap

---

## Milestone 0 🏗️ リポジトリ初期整備
- **Epic 0-1 Init repository & workflow**
  - [ ] **Add LICENSE** (OSS / proprietary) `kind:docs`
  - [ ] **Draft README.md** – 概要・論文リンク・図解1枚 `kind:docs`
  - [ ] **Create CONTRIBUTING.md** – コード規約 / PR フロー `kind:docs`
  - [ ] **Set up GitHub Projects** (board) `kind:meta`
  - [ ] **Enable branch-protection** (main → PR + 1 review) `kind:meta`
  - [ ] **Add Issue/PR templates** `kind:meta`

---

## Milestone 1 🛠️ 開発環境 & CI/CD
### Epic 1-1 Docker & dependency baseline `area:infra`
- [ ] Write **Dockerfile** (python 3.11 / cuda-base / pytorch / faiss-gpu)
- [ ] Pin **requirements.txt / poetry.lock**
- [ ] VS Code **dev-container** 設定 (optional)

### Epic 1-2 CI pipelines `area:infra`
- [ ] GitHub Actions: **lint + unit-test** (pytest + ruff/black)
- [ ] **CI cache** for Faiss & torch
- [ ] **CodeQL / secret-scan** on push `area:security`

---

## Milestone 2 🧠 コア層実装 – MVP
<details>
<summary>展開して見る</summary>

### Epic 2-1 L1 – Error Monitor (cerebellum)
- [ ] `predictive_coding.py` – 簡易自己回帰 + 誤差出力
- [ ] 閾値 **τ_err** を yaml config 化
- [ ] Unit test: 合成データで誤差トリガ確認

### Epic 2-2 L2 – Quantum-RAG (memory store) `area:l2`
- [ ] Build **Faiss IVF-PQ** indexer
- [ ] **Episode schema** (id, vector, C, metadata)
- [ ] Similarity **sim × C^γ** 実装
- [ ] CRUD API (add / search / delete / re-quantize)
- [ ] Unit tests (precision@k, re-index 速度)

### Epic 2-3 L3 – GNN + ΔGED/ΔIG + Conflict Score `area:l3`
- [ ] Graph schema (NetworkX → later PyG)
- [ ] **ΔGED** calculator (edit-distance POC)
- [ ] **ΔIG** metrics (NMI / entropy)
- [ ] Conflict Score module (rule-based stub)
- [ ] Reward **R(w₁,w₂,w₃)** & config
- [ ] Interface to L2 (C-value update / re-quantize signal)

### Epic 2-4 L4 – LLM interface
- [ ] Wrapper `llm.py` (OpenAI / local Llama.cpp)
- [ ] Prompt builder (context = e_recons + R)
- [ ] Streaming output & user-feedback hook

### Epic 2-5 Orchestrator – Layer pipeline `area:core`
- [ ] Message-bus (**AsyncIO** event loop)
- [ ] End-to-end **CLI demo** (“question → L4 answer”)
- [ ] Logging (traces JSONL + wandb optional)

</details>

---

## Milestone 3 📊 PoC 評価 & “閃き” 可視化
### Epic 3-1 QA dataset benchmark
- [ ] DL small **WikiQA / NQ-subset**
- [ ] Baseline (RAG-only) vs MVP
- [ ] **BLEU / BERTScore + ΔGED/ΔIG** プロット

### Epic 3-2 Synthetic graph spike-demo
- [ ] 10-node toy graph で ΔGED スパイク観測
- [ ] matplotlib 可視化スクリプト
- [ ] README gif 添付

---

## Milestone 4 🛡️ 安全設計 & ガードレール
### Epic 4-1 Hallucination & C-value decay
- [ ] 誤答指摘エピソードの C↓ 実装
- [ ] 人手 approval gate (CLI prompt)
- [ ] Unit test: C-value floor / ceiling

### Epic 4-2 Rate-limit & resource budget
- [ ] 再量子化頻度 **scheduler**
- [ ] GPU util logger + abort if load > threshold

---

## Milestone 5 🚀 拡張フェーズ
<details>
<summary>展開して見る</summary>

### Epic 5-1 Sleep-mode (refactor loop)
- [ ] Idle detector → 内部 loop
- [ ] Memory pruning policy
- [ ] Nightly cron job in CI

### Epic 5-2 Multimodal L0 (vision/audio)
- [ ] **CLIP embeddings** → L2
- [ ] Tiny **SSM audio encoder**
- [ ] Demo notebook

### Epic 5-3 Edge-device prototype
- [ ] Export **ONNX / TensorRT** L1-L3 core
- [ ] Jetson container build
- [ ] Latency benchmark (< 30 cm³ target)

</details>

---

## Milestone 6 📚 ドキュメント & コミュニティ
### Epic 6-1 Docs site with mkdocs-material
- [ ] `architecture.md` – 論文サマリ + 図
- [ ] `api_reference.md`
- [ ] `contributing_guide.md` (JP & EN)

### Epic 6-2 Research notes / open questions
- [ ] `docs/open_questions/` フォルダ作成
- [ ] Issue label management

### Epic 6-3 Outreach
- [ ] Enable **GitHub Discussions**
- [ ] README に “looking-for-collaborators” バッジ
- [ ] 初回 demo video (loom / youtube-unlisted)

---
