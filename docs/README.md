# InsightSpike-AI Documentation

Welcome to the InsightSpike-AI documentation! This directory contains all technical documentation, guides, and references for the project.

> **First time here (agent or collaborator)?** Read [MAP.md](./MAP.md) — the repository
> orientation guide (directory map, terminology traps, claims ledger, known debt).

## 📚 Documentation Structure

### 🚀 [Getting Started](./getting-started/)
New to InsightSpike? Start here!
- [Environment Setup](./getting-started/ENVIRONMENT_SETUP.md) - System requirements and installation
- [Setup Guide](./getting-started/setup_guide.md) - Detailed setup instructions
- [Quick Test](./getting-started/quick_test_discover.md) - Try the discover command

### 📖 [User Guide](./user-guide/)
Learn how to use InsightSpike effectively
- [July 2024 Features Quick Start](./user-guide/july_2024_features_quickstart.md) - Get started with the latest features
- [Configuration Guide](./user-guide/configuration_guide.md) - Configure InsightSpike for your needs
- [CLI Commands](./user-guide/cli_commands.md) - Complete command reference
- [Spike Commands Summary](./user-guide/spike_commands_summary.md) - Quick reference for spike commands
- [LLM Providers Guide](./user-guide/llm_providers_guide.md) - Setting up different LLM providers

### 🔧 [API Reference](./api-reference/)
Detailed API documentation for developers
- [API Summary](./api-reference/CORRECT_API_SUMMARY.md) - Core API overview
- [Detailed Documentation](./api-reference/DETAILED_DOCUMENTATION.md) - Comprehensive API reference

### 🏗️ [Architecture](./architecture/)
Understanding InsightSpike's design
- [Overview](./architecture/README.md) - System architecture introduction
- [Layer Architecture](./architecture/layer_architecture.md) - 4-layer brain-inspired design
- [Agent Types](./architecture/agent_types.md) - Available agent implementations
- [Directory Structure](./architecture/directory_structure.md) - Code organization
- [Configuration System](./architecture/configuration.md) - Config architecture
- [Data Management](./architecture/data_management_strategy.md) - Data handling strategies
- [MainAgent Behavior](./architecture/mainagent_behavior.md) - Core agent behavior
- [Multi-User Design](./architecture/multi_user_design.md) - Multi-user considerations

### 🧪 Experiments & Reproduction
Entry points for reproducing the paper experiments
- [Experiment Index](./EXPERIMENTS.md) - Where to find Maze / RAG experiments
- [Maze Navigation Spec](./MAZE_NAV_SPEC.md) - Maze environment and agent spec
- [Maze Metrics HOWTO](./HOWTO_maze_metrics.md) - How we compute maze metrics
- [Phase‑1 Overview](./phase1.md) - Phase‑1 (online) experiment design summary

### 📈 [Diagrams](./diagrams/)
Visual representations of system architecture
- [README](./diagrams/README.md) - Available diagrams
- Various system diagrams in PNG format

### 📄 Paper & Theory
Formal specification and theory background
- [geDIG Spec](./gedig_spec.md) - High‑level spec of the gauge
- **Paper v6.1 (current; claim–evidence consistency revision, 2026-06)**:
  JA `docs/paper/v6.1/geDIG_onegauge_improved_v6_1.pdf` / EN `docs/paper/v6.1/arxiv_en/geDIG_onegauge_improved_v6_1_en.pdf`
- Paper v6 (arXiv/Zenodo DOI-pinned): `docs/paper/v6/arxiv_en/geDIG_onegauge_improved_v6_en.pdf`
- Paper v5 (historical): `docs/paper/geDIG_onegauge_improved_v5.pdf`
- v7 plan (β₁ generalization, in progress): [paper/v7/plan.md](./paper/v7/plan.md) + [Tier 1 drafts](./paper/v7/draft_sections/)
- **Audits (2026-06, external review)**: [audits/](./audits/) — F-sign, PER metric, oracle routing ceiling
- **Pre-registrations**: [prereg/](./prereg/) — router duel (Stage A defeat recorded), maze sleep ablation

### 🗂️ Research Notes (Long‑Term / AGI)
Exploratory notes on AGI foundations, governance, and Phase‑2 ideas
- [AGI / Governance Notes](./research-notes/appendix/README.md)
- [Phase‑2 Offline Optimization (JA/EN)](./research-notes/appendix/phase2/phase2_offline_appendix_ja_en.md)

### 🖼️ [Images](./images/)
Documentation images and animations
- [Animations](./images/animations/) - Animated demonstrations

## 🔍 Finding What You Need

- **New Users**: Start with [Getting Started](./getting-started/)
- **Users**: Check [User Guide](./user-guide/) for daily usage
- **Developers**: See [API Reference](./api-reference/) and [Architecture](./architecture/)
- **Researchers**: Read the paper and theory docs:
  - Paper v6.1 (current): `docs/paper/v6.1/` (JA/EN)
  - Gauge spec: [geDIG Spec](./gedig_spec.md)
  - Audits and pre-registrations: [audits/](./audits/), [prereg/](./prereg/)
- **Contributors**: Review [CONTRIBUTING](./CONTRIBUTING.md) for ongoing work

## 📝 Documentation Standards

When contributing documentation:
1. Use clear, concise language
2. Include code examples where applicable
3. Keep files focused on a single topic
4. Update this index when adding new documents
5. Place files in the appropriate category directory

## 🔗 Quick Links

- [Main README](../README.md) - Project overview
- [Contributing Guide](./CONTRIBUTING.md) - How to contribute
- [Code of Conduct](./CODE_OF_CONDUCT.md) - Community guidelines
- [License](./LICENSE) - License information

---

*Last updated: July 2026*
