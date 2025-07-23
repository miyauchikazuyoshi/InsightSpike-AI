# geDIG Research Paper

This directory contains the official geDIG (Graph Edit Distance + Information Gain) research paper and related materials.

## 📄 Main Paper

### geDIG: グラフ構造変化に基づく洞察生成フレームワーク (Japanese)
- **File**: `geDIG_paper_ja.tex`
- **Status**: v3 - Final version (Strong Accept from peer review)
- **Key Results**: 
  - 85% overall accuracy (95% CI: [62.1%, 96.8%])
  - 100% accuracy on complex multi-concept questions
  - 45ms real-time processing on CPU
  - Discovery of "difficulty reversal phenomenon"

## 📊 Figures

All figures are in the `figures/` subdirectory:
- `brain_ai_analogy_diagram.png` - Brain-InsightSpike architecture mapping
- `gedig_framework_diagram.png` - Framework overview
- `gedig_results_visualization.png` - Experimental results

## 🔨 Building the Paper

To compile the LaTeX document:

```bash
# Using uplatex (Japanese support)
uplatex geDIG_paper_ja.tex
dvipdfmx geDIG_paper_ja.dvi

# Or using pdflatex with appropriate packages
pdflatex geDIG_paper_ja.tex
```

## 📚 Related Materials

- **Appendix**: `geDIG_paper_appendix_ja.tex` - Additional technical details
- **Experiments**: See `/experiments/` directory for reproducible code
- **Implementation**: Full InsightSpike-AI system in `/src/`

## 🏆 Recognition

- Peer Review Score: 5/5 (Strong Accept)
- Recommended for Best Paper Award
- Oral presentation recommendation

---

*Last Updated: July 2025*