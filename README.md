# EPRAF: Electrophysiology-Anesthesia RAG Assessment Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Paper:** EPRAF: A Proposed Hallucination-Aware Evaluation Framework for Electrophysiology Anesthesia RAG  
> **Submitted to:** NEJM AI — Datasets, Benchmarks, and Protocols  
> **Status:** Under review

---

## Overview

EPRAF is an open evaluation harness for assessing retrieval-augmented generation (RAG) systems in electrophysiology anesthesia clinical decision support. It addresses a critical gap: existing RAG evaluation frameworks cannot reliably detect the failure modes — hallucinated drug contraindications, cited misinformation, retrieval vocabulary mismatch — that matter most in this high-acuity subspecialty.

**EPRAF comprises four components:**

1. **50-query benchmark taxonomy** — stratified across four clinically motivated query categories (intraoperative emergency, drug interaction, device management, pre-procedure planning)
2. **Four-condition retrieval ablation protocol** — sparse-only, dense-only, hybrid RRF, hybrid + cross-encoder reranking
3. **AnchorScore annotation scheme** — 1–3 ordinal scale tied to retrievable evidence units; paraphrase-tolerant and sensitive to individual clinical claim accuracy
4. **NLI-based hallucination detection pipeline** — distinguishes *uncited hallucination* from *cited misinformation* (higher clinical severity)

---

## Key Findings (Pilot Simulation)

Three metric design requirements identified from single-seed pilot validation against a synthetic corpus:

| Finding | Observation | Implication |
|---------|-------------|-------------|
| Token F1 inadequacy | Range 0.232–0.237 (Δ=0.005) across conditions | Surface-form metrics provide no discriminative signal for clinical RAG |
| Hallucination floor effect | HallucinationRate = 0.0 uniformly across all conditions | Binary response-level detector useless; claim-level NLI required |
| Reranker dissociation | Hybrid+Rerank: MAP ↓ 30%, AnchorScore ↑ 0.063 | Precision-breadth trade-off requires domain-specific evaluation |

> ⚠️ All values are single-run software validation artifacts from synthetic data. No real clinical corpus, LLM, or retrieval system was evaluated.

---

## Repository Structure

```
EPRAF-public/
├── code/                    # Evaluation harness source
│   ├── main.py              # Entry point — runs the full evaluation pipeline
│   ├── evaluation.py        # Metric computation (NDCG, MAP, MRR, AnchorScore, F1)
│   ├── retrieval.py         # Four-condition retrieval (sparse, dense, hybrid, rerank)
│   ├── hallucination.py     # NLI-based claim-level hallucination detection
│   ├── data.py              # Benchmark query loader and synthetic corpus generator
│   ├── config.py            # Hyperparameter configuration
│   └── requirements.txt     # Python dependencies
├── charts/                  # Paper figures
│   ├── fig1_metric_heatmap.png
│   ├── fig2_anchor_score_comparison.png
│   ├── fig3_ablation_component.png
│   └── fig4_hallucination_rate.png
├── docs/                    # Manuscript and references
│   ├── manuscript.md        # Full paper (NEJM AI submission version)
│   └── references.bib       # BibTeX bibliography (32 verified references)
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run the full evaluation pipeline (pilot mode — synthetic corpus)
python code/main.py --mode pilot --seed 42

# Run a specific ablation condition
python code/main.py --mode pilot --condition hybrid_rerank
```

---

## Benchmark Query Categories

| Category | N | Example Query |
|---|---|---|
| Intraoperative emergency | 13 | Vasopressor selection during refractory hypotension in AF ablation under GA |
| Drug interaction | 13 | Dexmedetomidine use in patients with subcutaneous ICDs |
| Device management | 12 | Electrocautery reprogramming requirements for CRT-D patients |
| Pre-procedure planning | 12 | Anticoagulation bridging for patients with complex device histories |

---

## Evaluation Metrics

- **Retrieval:** NDCG@5, MAP, MRR, P@5
- **Response quality:** AnchorScore (primary), Token F1 (secondary / comparability)
- **Clinical safety:** HallucinationRate (uncited hallucination + cited misinformation)
- **Statistics:** Wilcoxon signed-rank, McNemar's test, Cohen's κ, Benjamini-Hochberg FDR

---

## Citation

If you use EPRAF in your research, please cite:

```bibtex
@article{epraf2026,
  title   = {EPRAF: A Proposed Hallucination-Aware Evaluation Framework 
             for Electrophysiology Anesthesia RAG},
  journal = {NEJM AI},
  year    = {2026},
  note    = {Under review},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## AI Disclosure

This framework was developed with the assistance of AutoResearchClaw, an automated research pipeline using Claude Sonnet (Anthropic, Inc.). All scientific content and methodology were reviewed and approved by the authors.
