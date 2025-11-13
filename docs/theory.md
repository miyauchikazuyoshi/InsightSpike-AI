---
layout: page
title: Theory — geDIG
permalink: /theory/
---

# Theory (Short)

## Unified gauge

We define a single gauge \(\mathcal{F}\) to decide When to accept a structural update in a dynamic knowledge graph:

\[\mathcal{F} = \Delta\mathrm{EPC}_{\mathrm{norm}} - \lambda\, \Delta\mathrm{IG}_{\mathrm{norm}},\quad \Delta\mathrm{IG}_{\mathrm{norm}} = \Delta H_{\mathrm{norm}} + \gamma\,\Delta\mathrm{SP}_{\mathrm{rel}}.\]

- \(\Delta\mathrm{EPC}_{\mathrm{norm}}\): normalized edit‑path cost of actually applied operations (operational)
- \(\Delta H_{\mathrm{norm}}\): normalized entropy change
- \(\Delta\mathrm{SP}_{\mathrm{rel}}\): relative change in average shortest path (signed)

### Sign conventions (recap)
- \(\Delta H_{\rm norm} = (H_{\rm after}-H_{\rm before})/\log K\) — order increases (entropy decreases) → negative
- \(\Delta\mathrm{SP}_{\rm rel}=(L_{\rm before}-L_{\rm after})/\max\{L_{\rm before},\varepsilon\}\) — path shortens → positive
- Smaller \(\mathcal{F}\) is better (accept when sufficiently small under gating)

## Two‑stage gating
We use 0‑hop Ambiguity Gate (AG) and multi‑hop Decision Gate (DG):
- AG (0‑hop): if uncertain, trigger retrieval
- DG (multi‑hop): confirm structural shortcut and commit

## PSZ / SLO
We adopt the Perfect Scaling Zone (PSZ) as an SLO for dynamic RAG under equal‑resources:

Targets (windowed over \(W\) queries):
\[ \mathrm{Acc}\ge 0.95,\quad \mathrm{FMR}\le 0.02,\quad P50_{\Delta\mathrm{lat}} \le 200\,\mathrm{ms}. \]

Deficit (lower is better):
\[ s_{\mathrm{PSZ}}=\max(0,0.95-\mathrm{Acc})+\max(0,\mathrm{FMR}-0.02)+\max\!\bigl(0,\tfrac{P50_{\Delta \mathrm{lat}}-200\,\mathrm{ms}}{200\,\mathrm{ms}}\bigr). \]

See the paper for full derivations and ablations.

