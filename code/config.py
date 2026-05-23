"""
Centralized hyperparameter and experiment configuration.
"""


class Config:
    # Dataset
    squad_v1_name = "rajpurkar/squad"
    squad_v2_name = "rajpurkar/squad_v2"
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    max_answer_length = 30
    n_best_size = 20
    null_score_diff_threshold = 0.0

    # EP Clinical Query Taxonomy
    ep_categories = [
        "arrhythmia_management",
        "device_indications",
        "perioperative_anticoagulation",
        "anesthetic_drug_interactions",
        "hemodynamic_emergencies",
    ]
    n_clinical_queries = 50
    queries_per_category = 10

    # EP Abbreviation Expansion Map
    ep_abbreviation_map = {
        "SVT": "supraventricular tachycardia",
        "PVC": "premature ventricular contraction",
        "VT": "ventricular tachycardia",
        "AF": "atrial fibrillation",
        "ICD": "implantable cardioverter defibrillator",
        "EP": "electrophysiology",
        "AFL": "atrial flutter",
        "WPW": "Wolff-Parkinson-White syndrome",
        "AVNRT": "atrioventricular nodal reentrant tachycardia",
        "CRT": "cardiac resynchronization therapy",
    }

    # Retrieval / RAG
    dense_model_name = "BAAI/bge-m3"
    cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_k1 = 1.5
    bm25_b = 0.75
    top_k_dense = 10
    top_k_sparse = 10
    top_k_rerank = 5
    rrf_k = 60
    chunk_size = 512
    chunk_overlap = 64
    context_compression_ratio = 0.5

    # Hallucination Detection
    evidence_anchor_scale = 5
    hallucination_flag_threshold = 1
    # Minimum NORMALIZED overlap fraction (0-1) for claim support.
    # A claim is supported if (shared_tokens / claim_tokens) >= this threshold.
    min_citation_overlap_fraction = 0.25
    # Minimum overlap tokens for cross-encoder reranker (integer token count)
    min_citation_overlap_tokens = 3

    # Models / Baselines
    bert_model_name = "bert-base-uncased"
    gpt2_model_name = "gpt2"
    bert_hidden_dim = 768
    gpt2_hidden_dim = 768

    # Training
    learning_rate = 3.0e-5
    batch_size = 8
    epochs = 3
    warmup_steps = 100
    weight_decay = 0.01
    max_grad_norm = 1.0
    seeds = [42, 123, 7]
    device = "cpu"  # default; overridden at runtime if CUDA available
    fp16 = False

    # Evaluation / Statistics
    alpha = 0.05
    power = 0.80
    accuracy_diff_threshold = 0.15
    latency_budget_seconds = 10.0
    mcnemar_continuity_correction = True

    # Compute Budget
    max_gpu = 1
    max_hours = 4
    time_budget_seconds = 14400

    # Output
    results_dir = "results"
    checkpoint_dir = "checkpoints"
    log_level = "INFO"

    def validate(self) -> list:
        """
        Validate configuration values. Returns list of error strings (empty = valid).
        Called at experiment start to catch misconfiguration early.
        """
        errors = []
        if self.queries_per_category * len(self.ep_categories) != self.n_clinical_queries:
            errors.append(
                f"queries_per_category ({self.queries_per_category}) x "
                f"n_categories ({len(self.ep_categories)}) != "
                f"n_clinical_queries ({self.n_clinical_queries})"
            )
        if not 0 < self.min_citation_overlap_fraction <= 1.0:
            errors.append("min_citation_overlap_fraction must be in (0, 1]")
        if self.top_k_rerank > self.top_k_dense or self.top_k_rerank > self.top_k_sparse:
            errors.append("top_k_rerank should be <= top_k_dense and top_k_sparse")
        if self.latency_budget_seconds <= 0:
            errors.append("latency_budget_seconds must be positive")
        return errors
