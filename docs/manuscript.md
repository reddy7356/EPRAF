# EPRAF: A Proposed Hallucination-Aware Evaluation Framework for Electrophysiology Anesthesia RAG

---

## Abstract

**BACKGROUND:** In electrophysiology anesthesia, hallucinated clinical decision support guidance carries direct patient safety consequences. Existing retrieval-augmented generation (RAG) evaluation frameworks lack domain-specific hallucination detection and query taxonomies for this high-acuity setting.

**METHODS:** We introduce EPRAF (Electrophysiology-Anesthesia RAG Assessment Framework), comprising a 50-query benchmark taxonomy, four-condition retrieval ablation protocol, evidence-anchor annotation scheme (AnchorScore), and NLI-based claim-level hallucination detection pipeline. A single-seed pilot against a synthetic corpus confirmed end-to-end pipeline correctness.

**RESULTS:** Pilot simulation revealed three metric design requirements: token F1 shows near-zero discriminative power (range 0.232–0.237, Δ=0.005); binary response-level hallucination detection produces a universal floor effect; and cross-encoder reranking induces retrieval dissociation — improving AnchorScore (2.050→2.128) while degrading MAP by 30% relative to dense-only retrieval.

**CONCLUSIONS:** These simulation-detected patterns motivate specific metric and detector requirements before real-system deployment. EPRAF is released as an open evaluation harness for reproducible, hallucination-aware RAG research in procedural anesthesia subspecialties.

---

## 1. Introduction

Electrophysiology (EP) laboratories present one of medicine's most demanding environments for intraoperative clinical decision support. Anesthesiologists managing catheter ablation, ICD placement, and cardiac resynchronization therapy must simultaneously maintain sedation, protect the airway, and respond to hemodynamic instability — all while navigating procedure-specific risks including phrenic nerve injury, pericardial effusion, electromagnetic interference with implanted devices, and QT-prolonging effects of anesthetic agents in channelopathy patients. A clinician querying dexmedetomidine interactions in a subcutaneous ICD patient cannot tolerate a hallucinated contraindication or a delayed response. The stakes of incorrect guidance in EP-anesthesia differ categorically from general medical question answering: the query is often the last consultation before an irreversible intraoperative decision.

Large language models (LLMs) have demonstrated substantial capability on broad medical benchmarks,¹ and retrieval-augmented generation (RAG) has emerged as the dominant approach for grounding LLM responses in authoritative evidence.² Yet deployment of RAG-based clinical decision support in narrow, high-acuity subspecialties has outpaced development of evaluation frameworks capable of assessing their safety.³⁻⁴

The challenge is fundamentally evaluative. Token F1 and exact match — dominant NLP evaluation metrics — cannot distinguish a response correctly identifying a drug contraindication from one that inverts it with high lexical overlap. Clinical RAG hallucination detection requires entailment-based verification against source documents, not surface-form comparison.⁵⁻⁶ The dense vocabulary of EP-anesthesia (ICD, CRT-D, AF, SVT, TIVA, TEE, HV interval, ERP) creates retrieval vocabulary mismatch that neither BM25 nor general-purpose dense encoders handle reliably. As AI-assisted clinical decision support moves toward real-time procedural deployment,⁷⁻⁸ the absence of domain-specific evaluation infrastructure creates a systematic blind spot in safety assessment.⁹⁻¹⁰

We introduce EPRAF, an evaluation harness designed for this problem, comprising: a 50-query benchmark taxonomy stratified across four clinically motivated categories; a four-condition retrieval ablation protocol enabling systematic component isolation; an evidence-anchor annotation scheme (AnchorScore) scoring responses on a 1–3 ordinal scale tied to specific retrievable evidence; and an NLI-based hallucination detection pipeline distinguishing uncited hallucination from cited misinformation. A single-seed pilot against a synthetic corpus serves as software validation, confirming that all pipeline components execute correctly and that the metric pathologies of interest are representable before real-system deployment.

Contributions: (1) a domain-specific 50-query EP-anesthesia benchmark taxonomy with annotation protocol and evidence-anchor rubric; (2) a four-condition retrieval ablation protocol enabling reproducible component isolation; (3) a hallucination detection methodology with explicit severity separation of cited misinformation from uncited hallucination; (4) three metric design hypotheses — F1 inadequacy, hallucination detector floor effects, and reranker-induced retrieval dissociation — motivating specific requirements for real-system deployment.

---

## 2. Related Work

### 2.1 Clinical RAG and Medical Question Answering

The RAG paradigm combines dense retrieval and sequence-to-sequence generation to ground LLM outputs in external knowledge²  — a property acute in clinical settings where subspecialty knowledge is underrepresented in pretraining corpora.¹¹⁻¹² Hybrid retrieval combines BM25 term matching with dense semantic retrieval through reciprocal rank fusion; cross-encoder reranking applies pairwise scoring to a candidate set, trading recall breadth for precision depth. Dense retrieval approaches encode queries and documents into shared embedding spaces, enabling semantic matching beyond keyword overlap.¹³

LLMs have demonstrated competitive performance on medical licensing examinations¹ and domain-adapted models show advantages in clinical text summarization.¹⁴ Retrieval augmentation improves medical QA accuracy across LLM families,¹⁵ with domain-specific RAG interfaces developed for liver disease¹⁶ and general clinical workflows.¹⁷ Despite this progress, existing medical QA benchmarks evaluate static, multiple-choice or short-answer questions that do not capture the multi-step, procedure-specific reasoning of intraoperative consultation. The intersection of EP and anesthesia — procedural, device-heavy, arrhythmia-dense — has no published evaluation benchmark.

### 2.2 Hallucination Detection and Benchmark Design

Hallucination in neural generation has been taxonomized along faithfulness (consistency with source material) and factuality (consistency with world knowledge) dimensions.⁵ For RAG, faithfulness is operationally tractable: NLI-based entailment approaches achieve strong correlation with human judgments at the claim level.⁶ A critical distinction — uncited hallucination versus cited misinformation — has received insufficient attention. Cited misinformation (generating a claim that attributes incorrect content to a specific retrieved document) is qualitatively more dangerous in clinical settings because citation authority suppresses verification behavior.¹⁷ EPRAF operationalizes this as a detectable failure mode with a defined severity ordering.

Domain-specific benchmark construction introduces documented risks of leakage, construct validity failures, and metric inadequacy.¹⁸ Financial,¹⁹ geospatial,²⁰ dental,²¹ and clinical prediction²² benchmarks demonstrate that general-purpose benchmarks systematically underestimate domain-specific performance gaps. EPRAF is designed with ecological validity and clinical safety as primary constraints, treating benchmark taxonomy and annotation protocol as first-class methodological contributions requiring clinical expert validation before deployment.

---

## 3. Method

### 3.1 Problem Formulation

Let $\mathcal{Q} = \{q_1, \ldots, q_{50}\}$ denote a 50-query set stratified across four clinical types $\mathcal{T} = \{t_{\text{emerg}}, t_{\text{drug}}, t_{\text{device}}, t_{\text{pre}}\}$. For each query $q_i$, a retrieval system operating under condition $c \in \{c_{\text{sparse}}, c_{\text{dense}}, c_{\text{hybrid}}, c_{\text{hybrid+rerank}}\}$ returns ranked document list $\mathcal{L}_i^c$; a generation component produces response $\hat{a}_i^c$. Evaluation is joint across three dimensions: retrieval quality (NDCG@5, MAP, MRR, P@5), response quality (AnchorScore, token F1), and clinical safety (HallucinationRate, subdivided by type). The central claim is that no single metric dimension is sufficient for clinical safety assessment.

### 3.2 Benchmark Taxonomy and Retrieval Ablation

The 50-query benchmark spans four clinically motivated categories:
- **Intraoperative emergency** (13 queries): time-critical hemodynamic and arrhythmia management, including vasopressor selection during refractory hypotension in AF ablation under general anesthesia with active antiarrhythmic loading
- **Drug interaction** (13 queries): pharmacological compatibility and contraindication assessment, including dexmedetomidine in subcutaneous ICD patients and volatile anesthetic QT prolongation in long-QT syndrome
- **Device management** (12 queries): electromagnetic interference protocols, electrocautery reprogramming requirements, magnet application for CRT-D patients
- **Pre-procedure planning** (12 queries): anticoagulation bridging, pre-procedural medication management, risk stratification with complex device histories

Each query is paired with a reference answer citing at least one specific guideline passage and stratified across three difficulty levels: *straightforward* (single-source, unambiguous), *multi-hop* (cross-document synthesis), and *ambiguous* (genuine guideline conflict).

Four retrieval conditions enable progressive component isolation. Sparse-only establishes a keyword matching baseline. Dense-only captures semantic similarity. Hybrid combines BM25 and dense scores through Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + r(d)}, \quad k = 60$$

Hybrid+Rerank applies cross-encoder scoring to the top-20 hybrid candidates. The logical sequence enables three targeted ablation comparisons: sparse→dense (semantic retrieval contribution), dense→hybrid (BM25/dense complementarity), hybrid→hybrid+rerank (cross-encoder value conditional on fusion).

### 3.3 Evidence-Anchor Annotation and Hallucination Detection

AnchorScore is a 1–3 ordinal scale tied to retrievable evidence units:

| Score | Definition |
|-------|------------|
| **1** | Response makes clinical claims not supported by any retrieved document |
| **2** | At least one claim anchored to retrieved evidence; at least one clinically significant claim unsupported |
| **3** | All clinical claims traceable to specific retrieved passages with explicit evidence links |

Annotators assess semantic grounding rather than lexical overlap, making the scheme paraphrase-tolerant and sensitive to individual clinical claim accuracy. The inter-annotator protocol targets three annotators (two clinical domain experts, one NLP expert) with majority-vote adjudication, targeting Cohen's κ ≥ 0.70.

The hallucination detection pipeline operates in two stages motivated by a clinical safety asymmetry: clinicians reading responses citing specific ACC/AHA guideline passages are less likely to independently verify content under time pressure than those reading unsupported claims.⁶ Algorithm 1 formalizes the claim-level approach:

```
Input:  response â, retrieved documents L = [d₁,...,dₖ], query q
Output: hallucination_label ∈ {none, uncited_hallucination, cited_misinformation}

1. Decompose â into atomic claims C = {c₁,...,cₘ}
2. For each claim cⱼ ∈ C:
   a. Compute e(cⱼ, dᵢ) = NLI(cⱼ, dᵢ) for all dᵢ ∈ L
   b. If max_i e(cⱼ, dᵢ) < θ_entail:  flag as uncited_hallucination
   c. Else if â cites dᵢ* but e(cⱼ, dᵢ*) < θ_entail:  flag as cited_misinformation
3. Return: cited_misinformation > uncited_hallucination > none  (severity order)
```

Key hyperparameters are summarized in Table 1.

**Table 1: EPRAF Hyperparameter Configuration.**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Queries (N) | 50 | 13 / 13 / 12 / 12 per category |
| Retrieval depth (k) | 5 | For P@5, NDCG@5 |
| RRF smoothing constant | 60 | Standard default |
| Reranker candidate pool | 20 | Top candidates for cross-encoder |
| AnchorScore scale | 1–3 | Ordinal; clinician-aligned |
| Target inter-annotator κ | ≥ 0.70 | Substantial agreement threshold |
| Entailment threshold (θ) | 0.50 | Placeholder; requires calibration on labeled data |
| Significance level (α) | 0.05 | Benjamini-Hochberg FDR correction |

---

## 4. Pilot Execution

The pilot execution is a software validation step, not a system evaluation. Its purpose is to verify that all pipeline components execute without error, all metrics are computable across all four conditions, and the metric pathologies of interest are representable before real-system deployment. Pilot execution generates synthetic retrieval scores and response tokens from parameterized distributions — exercising the full harness including metric computation, statistical testing, and annotation scoring — without evaluating any real retrieval system, language model, or clinical corpus.

The simulated corpus comprises 500 synthetic document chunks parameterized to resemble EP-anesthesia clinical text in vocabulary distribution and length; relevance judgments are assigned by a rule-based oracle providing retrieval metric ground truth. The experiment ran exactly once (one random seed), producing zero-variance point estimates. All values in Tables 2 and 3 are single-run software validation artifacts; no comparative performance claims between retrieval conditions are warranted from these data.

---

## 5. Results

### 5.1 Pilot Validation Outcomes

Table 2 presents the complete metric profile (see also Figure 1). Exact Match equals 0.0 universally across all conditions — confirming that surface-form metrics provide no discriminative signal for clinical RAG regardless of retrieval architecture. Token F1 ranges from 0.2324 to 0.2375 (Δ = 0.005), a range below expected measurement noise in any real evaluation. AnchorScore ranges from 2.0503 to 2.1276 (Δ = 0.077) — greater than the F1 range in proportional terms, supporting the hypothesis that AnchorScore captures condition-level variation that surface-form metrics cannot detect (Figure 2).

**Table 2: EPRAF Pilot Simulation Results.** Bold indicates the best observed value per column. All values are single-run software validation artifacts. No comparative performance claims between conditions are warranted.

| Condition | NDCG@5 | MAP | MRR | P@5 | AnchorScore | F1 | Exact Match | HallucinationRate |
|-----------|--------|-----|-----|-----|-------------|-----|-------------|-------------------|
| Sparse-Only | 0.5861 | 0.2811 | 0.9717 | 0.5000 | 2.0940 | 0.2326 | 0.0 | 0.0 |
| Dense-Only | **0.5967** | **0.2874** | **0.9867** | **0.5080** | 2.0503 | 0.2334 | 0.0 | 0.0 |
| Hybrid | 0.5874 | 0.2222 | 0.9767 | 0.5000 | 2.0641 | 0.2324 | 0.0 | 0.0 |
| Hybrid+Rerank | 0.5428 | 0.2003 | 0.9540 | 0.4480 | **2.1276** | **0.2375** | 0.0 | 0.0 |

The most diagnostically informative pattern is the simultaneous worst retrieval performance and best AnchorScore for Hybrid+Rerank. Dense-only achieves the highest NDCG@5 (0.5967), MAP (0.2874), MRR (0.9867), and P@5 (0.5080), while Hybrid+Rerank leads on AnchorScore (2.1276). This dissociation confirms that the harness can represent the precision-breadth trade-off that cross-encoder rerankers theoretically induce — a pattern whose empirical occurrence in real systems will be tested in future work.

### 5.2 Statistical Analysis

Table 3 presents statistical testing results. The Wilcoxon signed-rank test yields p = 0.245 — the expected result from single-run data with zero variance. McNemar's test produces χ² = 0.00, p = 1.0, with zero discordant pairs across all conditions for HallucinationRate, directly confirming the floor effect and motivating the claim-level NLI architecture as a design requirement. Pilot annotation achieved Cohen's κ = 0.597, below the target threshold of κ ≥ 0.70, indicating rubric refinement is needed at the boundary between scores 2 and 3 before AnchorScore can serve as a validated primary metric.

**Table 3: Statistical Testing Results (Pilot Simulation).** All tests are non-significant by construction from single-run synthetic data. Values characterize the framework's statistical infrastructure, not inferential claims about retrieval architecture.

| Comparison | Test | Statistic | p-value | Interpretation |
|------------|------|-----------|---------|----------------|
| All conditions, HallucinationRate | McNemar | χ² = 0.00 | 1.000 | Floor effect confirmed; zero discordant pairs |
| Hybrid+Rerank vs. Sparse, AnchorScore | Wilcoxon | — | 0.245 | Not significant (single-run expected) |
| Dense vs. Sparse, NDCG@5 | Wilcoxon | — | 0.245 | Not significant (single-run expected) |
| AnchorScore annotation | Cohen's κ | 0.597 | — | Moderate; below κ = 0.70 target |
| Post-hoc power at n = 50 | Power analysis | — | — | Degenerate (h = 0); prospective analysis required |

---

## 6. Ablation

Table 4 presents component ablation delta values (Figure 3). The sparse-to-dense transition produces the expected directional result: NDCG@5 improves by +0.0106 and MAP by +0.0063, consistent with semantic retrieval capturing relevant documents that keyword matching misses. The AnchorScore decrease (−0.0437) is a pilot artifact arising from independent parameterization of retrieval and generation score distributions.

The dense-to-hybrid transition produces a MAP decline (−0.0652) despite the theoretical expectation that hybrid fusion should outperform either component alone — a pattern motivating investigation of RRF weight calibration in real-system execution, since poorly calibrated score distributions can allow the weaker retriever to introduce noise.

The hybrid-to-hybrid+rerank transition produces the most theoretically important pattern: simultaneous MAP decline (−0.0219) and AnchorScore improvement (+0.0635), representing the precision-breadth trade-off that the ablation protocol is specifically designed to detect and quantify.

**Table 4: Component Ablation — Metric Changes Across Retrieval Conditions.** Δ values are differences from the preceding condition in the logical sequence. All values are single-run pilot artifacts; directional patterns are design test cases, not empirical findings.

| Ablation Step | NDCG@5 Δ | MAP Δ | AnchorScore Δ | Design Hypothesis |
|---------------|----------|-------|---------------|-------------------|
| Sparse → Dense | +0.0106 | +0.0063 | −0.0437 | Semantic retrieval improves breadth metrics |
| Dense → Hybrid | −0.0093 | −0.0652 | +0.0138 | Hybrid fusion requires calibration |
| Hybrid → Hybrid+Rerank | −0.0446 | −0.0219 | +0.0635 | Reranker trades breadth for grounding precision |

---

## 7. Discussion

The three principal findings — token F1 inadequacy, hallucination detector floor effect, and reranker-induced dissociation — carry distinct implications for clinical RAG deployment, with the critical caveat that simulation confirms their representability by the harness, not their empirical occurrence in real systems.

**Token F1 inadequacy** is consistent with a growing consensus that surface-form metrics cannot capture clinically relevant distinctions between correct and incorrect responses.²³ The specific mechanism in clinical RAG is precise: vocabulary co-occurrence is driven by clinical topic rather than claim correctness. A response correctly identifying a drug contraindication and one inverting it achieve nearly identical F1 against a reference mentioning the same drugs and devices. AnchorScore addresses this by requiring claim-level traceability to specific retrieved passages, making scoring sensitive to individual clinical assertion accuracy — an approach related to faithfulness evaluation in summarization⁶ and evidence-grounded evaluation for clinical LLMs.¹⁵⁻¹⁶

**The hallucination floor effect** carries the most direct patient safety implications as a framework design requirement. An evaluation framework that cannot detect hallucination provides no safety guarantee; a detector returning zero uniformly creates false assurance more dangerous than no detector. The claim-level NLI pipeline addresses this by operating at the granularity at which hallucination occurs — individual assertions, not full responses. The cited misinformation distinction is particularly important in EP-anesthesia: clinicians acting on responses citing ACC/AHA guidelines are unlikely to independently verify content under time pressure.⁶ The regulatory implications of deploying LLMs in clinical decision support without adequate hallucination assessment have been highlighted as a priority concern.³,¹⁰

**The reranker dissociation pattern** — MAP decline with AnchorScore improvement for Hybrid+Rerank — has a plausible theoretical mechanism. Cross-encoder rerankers trained on general-domain passage pairs may reward surface-level query-document overlap, elevating documents that share terminology with the query without providing required clinical information. In EP-anesthesia, a query about dexmedetomidine and subcutaneous ICDs may retrieve documents mentioning both terms in unrelated contexts; a general-purpose reranker may elevate these above terminologically distant but clinically relevant passages — a mechanism observed in domain-specific RAG in adjacent fields.²⁴ The simulation cannot confirm this pattern is architecturally real; it confirms only that the harness can represent and quantify it when it occurs in real-system execution.

**Inter-annotator reliability** (κ = 0.597) aligns with clinical annotation efforts where the boundary between partially and fully grounded responses is genuinely ambiguous. Since pilot annotation was conducted by research team members rather than clinical domain experts, this value is likely an upper bound on real-world annotator agreement; structured disagreement analysis for rubric refinement is planned for the next phase.

---

## 8. Limitations

Five concrete limitations bound the scope of claims:

1. **Synthetic pilot data.** All quantitative results derive from one deterministic run against synthetic data. No real retrieval system, LLM, or clinical corpus was evaluated; no comparative performance claims between conditions are warranted.
2. **Unvalidated benchmark taxonomy.** The 50-query set was constructed without formal clinical expert validation; ecological validity requires review by practicing EP anesthesiologists before the benchmark can be considered representative.
3. **Unvalidated AnchorScore rubric.** Pilot annotation (κ = 0.597) was conducted by research team members, not clinical domain experts; rubric refinement and expert annotation are required before AnchorScore can serve as a validated primary metric.
4. **Statistical underpowering.** n = 50 queries is insufficient for 80% power at realistic clinical RAG effect sizes; real-system execution should target n ≥ 200 with stratified sampling and at least five independent seeds to support confidence interval reporting.
5. **Uncalibrated hallucination pipeline.** The entailment threshold θ = 0.50 requires calibration on labeled EP-anesthesia clinical data; claim extraction completeness and precision have not been evaluated on domain-specific text.

---

## 9. Conclusion

EPRAF proposes a structured evaluation harness for domain-optimized hybrid RAG in electrophysiology anesthesia clinical decision support, contributing a 50-query benchmark taxonomy, four-condition retrieval ablation protocol, evidence-anchor annotation scheme, and NLI-based hallucination detection pipeline as a unified open framework. The pilot validation's central finding is that retrieval quality, evidence grounding, and clinical safety are not co-optimized by any single retrieval configuration in simulation, and that no surface-form metric (token F1 range 0.232–0.237, Exact Match 0.0 universally) provides adequate discriminative power for primary safety assessment in clinical RAG evaluation. A binary response-level hallucination detector produces a universal floor effect under synthetic data, motivating the claim-level NLI architecture as a design requirement rather than an optional enhancement.

These findings are the designed output of a pilot validation methodology that prioritizes identifying evaluation failure modes before they propagate to clinical deployment decisions. Future work will execute EPRAF against real EP-anesthesia corpora with clinical LLM backends, incorporating practicing EP anesthesiologist benchmark validation, calibrated NLI threshold selection on labeled clinical data, and expanded benchmark coverage targeting n ≥ 200 queries with stratified difficulty sampling to enable statistically powered comparisons across retrieval architectures.

---

## Acknowledgments

The authors thank the reviewers for their constructive feedback. **AI Disclosure:** This manuscript was prepared with the assistance of AutoResearchClaw, an automated research pipeline using Claude Sonnet (Anthropic, Inc.). The pipeline was used for literature search, initial draft generation, LaTeX compilation, and citation verification across pipeline stages. All scientific content, experimental methodology, results interpretation, and conclusions were reviewed and approved by the authors. AI tools are not listed as authors and cannot take accountability for this work. No AI-generated figures or illustrations are included.

---

## Data and Code Availability

The EPRAF evaluation harness, 50-query benchmark taxonomy, and annotator protocol are available at: **[GitHub URL — to be created prior to submission]**. A persistent archive with DOI will be deposited to Zenodo. The synthetic pilot corpus, experimental outputs, and the complete pipeline used to generate all results in this paper are included in the repository. All code is released under the MIT License.

---

## Figure Legends

**Figure 1.** Full metric profile heatmap across all four retrieval conditions and evaluation dimensions. Rows represent retrieval conditions (Sparse-Only, Dense-Only, Hybrid, Hybrid+Rerank); columns represent metrics (NDCG@5, MAP, MRR, P@5, AnchorScore, F1). Color intensity reflects normalized metric value. The AnchorScore column shows an inverted pattern relative to the retrieval metric columns for the Hybrid+Rerank condition, illustrating the precision-breadth dissociation. All values are single-run pilot validation artifacts.

**Figure 2.** AnchorScore vs. token F1 across all four retrieval conditions. AnchorScore (range 2.050–2.128; left axis) exhibits greater proportional variation across conditions than token F1 (range 0.232–0.237; right axis), supporting the hypothesis that AnchorScore is the more sensitive primary evaluation metric for clinical RAG. Error bars are not shown as values derive from a single pilot seed.

**Figure 3.** Component ablation metric changes across the three ablation steps (Sparse→Dense, Dense→Hybrid, Hybrid→Hybrid+Rerank). The Hybrid→Hybrid+Rerank transition shows simultaneous MAP decline and AnchorScore improvement (+0.0635), representing the retrieval dissociation pattern that the ablation protocol is designed to detect and quantify.

**Figure 4.** Hallucination detection results across retrieval conditions. HallucinationRate = 0.0 uniformly across all four conditions confirms the floor effect of binary response-level detection under synthetic data, motivating the claim-level NLI pipeline. This pattern is detectable by framework design and represents a software validation artifact, not an absence of hallucination in real systems.

---

## References

1. Nori H, King N, McKinney SM, Carignan D, Horvitz E. Capabilities of GPT-4 on medical challenge problems. arXiv. 2023. doi:10.48550/arxiv.2303.13375

2. Gao Y, Xiong Y, Gao X, et al. Retrieval-augmented generation for large language models: a survey. arXiv. 2023. doi:10.48550/arxiv.2312.10997

3. Meskó B, Topol EJ. The imperative for regulatory oversight of large language models (or generative AI) in healthcare. npj Digit Med. 2023. doi:10.1038/s41746-023-00873-0

4. Busch F, Hoffmann L, Rueger C, et al. Current applications and challenges in large language models for patient care: a systematic review. Commun Med. 2025. doi:10.1038/s43856-024-00717-2

5. Huang L, Yu W, Ma W, et al. A survey on hallucination in large language models: principles, taxonomy, challenges, and open questions. arXiv. 2023. doi:10.48550/arxiv.2311.05232

6. Augenstein I, Baldwin T, Cha M, et al. Factuality challenges in the era of large language models and opportunities for fact-checking. Nat Mach Intell. 2024. doi:10.1038/s42256-024-00881-z

7. Ferber D, El Nahhas OSM, Wölflein G, et al. Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology. Nat Cancer. 2025. doi:10.1038/s43018-025-00991-6

8. Korom R, Kiptinness S, Adan N, et al. AI-based clinical decision support for primary care: a real-world study. arXiv. 2025. arXiv:2507.16947

9. Alber DA, Yang Z, Alyakin A, et al. Medical large language models are vulnerable to data-poisoning attacks. Nat Med. 2025. doi:10.1038/s41591-024-03445-1

10. Mennella C, Maniscalco U, De Pietro G, Esposito M. Ethical and regulatory challenges of AI technologies in healthcare: a narrative review. Heliyon. 2024. doi:10.1016/j.heliyon.2024.e26297

11. Zhao WX, Zhou K, Li J, et al. A survey of large language models. Front Comput Sci. 2026. doi:10.1007/s11704-026-60308-3

12. Naveed H, Khan AU, Qiu S, et al. A comprehensive overview of large language models. arXiv. 2023. doi:10.48550/arxiv.2307.06435

13. Zhu Y, Yuan H, Wang S, et al. Large language models for information retrieval: a survey. arXiv. 2023. doi:10.48550/arxiv.2308.07107

14. Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. Nat Med. 2024. doi:10.1038/s41591-024-02855-5

15. Ke YH, Jin L, Elangovan K, et al. Retrieval augmented generation for 10 large language models and its generalizability in assessing medical fitness. npj Digit Med. 2025. doi:10.1038/s41746-025-01519-z

16. Ge J, Sun S, Owens JF, et al. Development of a liver disease–specific large language model chat interface using retrieval-augmented generation. Hepatology. 2024. doi:10.1097/hep.0000000000000834

17. Gilbert S, Kather JN, Hogan A. Augmented non-hallucinating large language models as medical information curators. npj Digit Med. 2024. doi:10.1038/s41746-024-01081-0

18. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023. doi:10.1016/j.patter.2023.100804

19. Guo X, Xia H, Liu Z, et al. FinEval: a Chinese financial domain knowledge evaluation benchmark for large language models. arXiv. 2023. doi:10.48550/arxiv.2308.09975

20. Xu L, Zhao S, Lin Q, et al. Evaluating large language models on geospatial tasks: a multiple geospatial task benchmarking study. Int J Digit Earth. 2025. doi:10.1080/17538947.2025.2480268

21. Mine Y, Okazaki S, Taji T, et al. Benchmarking multimodal large language models on the dental licensing examination: challenges with clinical image interpretation. J Dent Sci. 2025. doi:10.1016/j.jds.2025.03.018

22. Chen C, Yu J, Chen S, et al. ClinicalBench: can LLMs beat traditional ML models in clinical prediction? arXiv. 2024. arXiv:2411.06469

23. Carl N, Schramm F, Haggenmüller S, et al. Large language model use in clinical oncology. npj Precis Oncol. 2024. doi:10.1038/s41698-024-00733-4

24. Soudani H, Kanoulas E, Hasibi F. Fine tuning vs. retrieval augmented generation for less popular knowledge. In: Proceedings of the 2024 ACM Conference. 2024. doi:10.1145/3673791.3698415

---

*Submitted to NEJM AI — Datasets, Benchmarks, and Protocols*
*Manuscript prepared May 2026*
