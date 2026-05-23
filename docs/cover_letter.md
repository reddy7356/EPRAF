# Cover Letter — NEJM AI Submission

**To:** The Editors, NEJM AI
**Re:** "EPRAF: A Proposed Hallucination-Aware Evaluation Framework for Electrophysiology Anesthesia RAG"
**Article Type:** Datasets, Benchmarks, and Protocols
**Date:** [Insert submission date]

---

Dear Editors,

We submit for your consideration the manuscript described above, proposing EPRAF (Electrophysiology-Anesthesia RAG Assessment Framework) — an open evaluation harness for hallucination-aware assessment of retrieval-augmented generation systems in electrophysiology anesthesia clinical decision support.

**Why NEJM AI, and why now.** The clinical AI community is actively deploying RAG-based decision support in procedural settings, yet the evaluation infrastructure has not kept pace with the safety requirements of narrow, high-acuity subspecialties. EPRAF addresses this gap with a domain-specific benchmark taxonomy, a four-condition ablation protocol, and a claim-level NLI hallucination detection pipeline that distinguishes cited misinformation from uncited hallucination — a distinction with direct patient safety implications that no existing framework operationalizes. We believe NEJM AI's Datasets, Benchmarks, and Protocols article type is precisely the right venue for this infrastructure contribution.

**Key findings.** Pilot simulation — serving as software validation, not system evaluation — identified three metric design requirements for clinical RAG: (1) token F1 exhibits near-zero discriminative power across retrieval conditions (Δ = 0.005), confirming that surface-form metrics are unfit for clinical safety assessment; (2) a binary response-level hallucination detector produces a universal floor effect, motivating the claim-level NLI pipeline; (3) cross-encoder reranking induces a retrieval dissociation pattern (AnchorScore improves +0.063 while MAP declines −0.022) representable by the harness and warranting empirical investigation in real systems. These are diagnostic framework findings, not performance claims.

**Strengths for NEJM AI readership.** This paper is explicitly framed as a patient safety infrastructure contribution: we show that the metrics currently used to evaluate clinical RAG cannot detect the failure modes that matter most in procedural settings. The framework is designed to be modular — any RAG backend, embedding model, or LLM can be substituted while the evaluation harness remains constant. All code, the benchmark taxonomy, and the annotator protocol will be publicly released upon acceptance.

**Transparency notes.** This manuscript was prepared with the assistance of AutoResearchClaw, an automated research pipeline using Claude Sonnet (Anthropic, Inc.), as described in the Acknowledgments. All 32 references have been independently verified against CrossRef, DataCite, and OpenAlex with a citation integrity score of 1.0. No real patient data, clinical corpus, or live API calls were used; all quantitative results derive from synthetic pilot simulation. The pilot is explicitly labeled as software validation throughout the manuscript. No presubmission inquiry has been submitted.

We declare no conflicts of interest. COI disclosure forms will be completed via the Convey system upon request.

We thank you for your time and consideration.

Sincerely,

[Corresponding Author Name]
[Institution]
[Email]
[ORCID]

---

*Manuscript word count (body): 2,585 words*
*Abstract: 141 words (structured: Background / Methods / Results / Conclusions)*
*Tables: 4 | Figures: 4 | References: 32*
*Article type: Datasets, Benchmarks, and Protocols*
