# Contributing to InsightSpike-AI

👍 **We welcome research collaboration and code review!**  
❗ This project is released under InsightSpike Open RAIL-M (research-only).  
Commercial contributions require written consent.

## Pull-Request Checklist
- [ ] My change does **not** violate “Additional Use Restrictions” in `LICENSE`.
- [ ] Code passes `ruff` / `black` / `pytest`.
- [ ] Added or updated unit tests if applicable.
- [ ] Docs / examples updated.
- [ ] **All public functions and classes have type hints.**
- [ ] **Code is formatted with `black` and imports are sorted with `isort`.**

## Branching flow
main (protected) ← develop ← feature/xxx

1. `git checkout -b feature/my-fix`
2. Commit (+ conventional-commit prefix, e.g. `feat:` `fix:` `docs:`)
3. Open PR → 1 reviewer approval → squash & merge

## Setting up locally

```bash
poetry install --with dev
pre-commit install
pytest
```

### 2-3 Issue & PR テンプレ (GUI 内で OK)

- **Bug report**  
  - steps to reproduce / expected vs actual / logs
- **Feature request (research-note)**  
  - problem statement / proposed solution / related papers
- **Pull request**  
  - checklist（上と同じ）＋ “I confirm this PR is research-only ✅”

---




