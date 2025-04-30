---
name: PULL_REQUEST_TEMPLATE.md
about: checklist
title: ''
labels: ''
assignees: miyauchikazuyoshi

---

# Contributing to InsightSpike-AI

👍 **We welcome research collaboration and code review!**  
❗ This project is released under InsightSpike Open RAIL-M (research-only).  
Commercial contributions require written consent.

## Pull-Request Checklist
- [ ] My change does **not** violate “Additional Use Restrictions” in `LICENSE`.
- [ ] Code passes `ruff` / `black` / `pytest`.
- [ ] Added or updated unit tests if applicable.
- [ ] Docs / examples updated.

## Branching flow
text<br>main (protected) ← develop ← feature/xxx<br>


1. `git checkout -b feature/my-fix`
2. Commit (+ conventional-commit prefix, e.g. `feat:` `fix:` `docs:`)
3. Open PR → 1 reviewer approval → squash & merge
mai
## Setting up locally

poetry install --with dev
pre-commit install
pytest

### 2-3 Issue & PR テンプレ (GUI 内で OK)

- **Bug report**  
  - steps to reproduce / expected vs actual / logs
- **Feature request (research-note)**  
  - problem statement / proposed solution / related papers
- **Pull request**  
  - checklist（上と同じ）＋ “I confirm this PR is research-only ✅”

---

## 3. 次のアクション

| 優先 | やること |
|------|----------|
| ★ | 上の README / CONTRIBUTING をコピペ → *Commit* |
| ★ | Settings › Features → “Set up templates” → 3 種類を GUI で作成 |
| ★ | Settings › Branches → “Add rule” → `main` → Require 1 review |
| ☆ | Projects タブ → *New project* → **InsightSpike Roadmap** → 列：To do ／ In-progress ／ Review ／ Done |

これで **Milestone 0 が完全終了**。  
さらなるブレークダウンや次フェーズ（Docker & CI）に進むタイミングでまた声をかけてください！
