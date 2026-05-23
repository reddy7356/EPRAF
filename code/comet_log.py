"""
comet_log.py — Upload EPRAF pilot results, figures, and artifacts to Comet ML.

Usage:
    COMET_API_KEY=your_key python comet_log.py

Or set COMET_API_KEY in your shell environment first, then:
    python comet_log.py

After running, the script prints a shareable public URL for the experiment.
Make the project public in Comet UI: Settings → Project → Visibility → Public.
"""

import os
import pathlib
import tempfile

# Suppress matplotlib/fontconfig cache warnings
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

# ── Comet ML ──────────────────────────────────────────────────────────────────
from comet_ml import Experiment, Artifact

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent.parent   # EPRAF-public/
CHARTS_DIR = ROOT / "charts"
CODE_DIR   = ROOT / "code"
DOCS_DIR   = ROOT / "docs"

# ── Pilot simulation results (from paper Tables 2, 3, 4) ──────────────────────
PILOT_RESULTS = {
    "sparse_only": {
        "ndcg_at_5":        0.5861,
        "map":              0.2811,
        "mrr":              0.9717,
        "precision_at_5":   0.5000,
        "anchor_score":     2.0940,
        "token_f1":         0.2326,
        "exact_match":      0.0,
        "hallucination_rate": 0.0,
    },
    "dense_only": {
        "ndcg_at_5":        0.5967,
        "map":              0.2874,
        "mrr":              0.9867,
        "precision_at_5":   0.5080,
        "anchor_score":     2.0503,
        "token_f1":         0.2334,
        "exact_match":      0.0,
        "hallucination_rate": 0.0,
    },
    "hybrid": {
        "ndcg_at_5":        0.5874,
        "map":              0.2222,
        "mrr":              0.9767,
        "precision_at_5":   0.5000,
        "anchor_score":     2.0641,
        "token_f1":         0.2324,
        "exact_match":      0.0,
        "hallucination_rate": 0.0,
    },
    "hybrid_rerank": {
        "ndcg_at_5":        0.5428,
        "map":              0.2003,
        "mrr":              0.9540,
        "precision_at_5":   0.4480,
        "anchor_score":     2.1276,
        "token_f1":         0.2375,
        "exact_match":      0.0,
        "hallucination_rate": 0.0,
    },
}

ABLATION_DELTAS = {
    "sparse_to_dense":        {"ndcg_delta": +0.0106, "map_delta": +0.0063, "anchor_score_delta": -0.0437},
    "dense_to_hybrid":        {"ndcg_delta": -0.0093, "map_delta": -0.0652, "anchor_score_delta": +0.0138},
    "hybrid_to_hybrid_rerank":{"ndcg_delta": -0.0446, "map_delta": -0.0219, "anchor_score_delta": +0.0635},
}

STATISTICAL_RESULTS = {
    "mcnemar_chi2":         0.00,
    "mcnemar_p_value":      1.0,
    "wilcoxon_p_value":     0.245,
    "cohen_kappa":          0.597,
    "kappa_target":         0.70,
    "n_queries":            50,
    "n_conditions":         4,
    "pilot_seeds":          1,
    "hallucination_floor_effect_confirmed": True,
    "f1_discriminative_range": 0.005,
    "anchor_score_range":   0.077,
}

FRAMEWORK_PARAMS = {
    "n_queries":            50,
    "query_categories":     4,
    "retrieval_depth_k":    5,
    "rrf_k":                60,
    "reranker_pool":        20,
    "anchor_score_scale":   "1-3",
    "target_kappa":         0.70,
    "entailment_threshold": 0.50,
    "alpha":                0.05,
    "fdr_correction":       "Benjamini-Hochberg",
    "pilot_mode":           "synthetic_validation",
}


def main():
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "COMET_API_KEY not set.\n"
            "Run:  export COMET_API_KEY=your_key_here   then retry."
        )

    print("🚀  Starting Comet ML experiment: EPRAF-pilot-v1")

    # workspace is omitted — Comet auto-uses the account tied to the API key
    experiment = Experiment(
        api_key=api_key,
        project_name="epraf",
        auto_metric_logging=False,
        auto_param_logging=False,
        log_code=False,
    )

    experiment.set_name("EPRAF-pilot-v1")
    experiment.add_tags(["pilot", "synthetic", "validation", "nejm-ai", "clinical-rag",
                         "hallucination-detection", "electrophysiology", "anesthesia"])

    # ── 1. Framework hyperparameters ─────────────────────────────────────────
    print("  Logging framework parameters...")
    experiment.log_parameters(FRAMEWORK_PARAMS)

    # ── 2. Per-condition pilot metrics ───────────────────────────────────────
    print("  Logging pilot metrics per condition...")
    condition_order = ["sparse_only", "dense_only", "hybrid", "hybrid_rerank"]
    condition_labels = {
        "sparse_only":   "Sparse-Only (BM25)",
        "dense_only":    "Dense-Only",
        "hybrid":        "Hybrid (RRF)",
        "hybrid_rerank": "Hybrid + Cross-Encoder Rerank",
    }

    for step, condition in enumerate(condition_order):
        metrics = PILOT_RESULTS[condition]
        label   = condition_labels[condition]
        for metric_name, value in metrics.items():
            experiment.log_metric(
                name=f"{condition}/{metric_name}",
                value=value,
                step=step,
            )
        print(f"    ✓ {label}")

    # ── 3. Summary / headline metrics ────────────────────────────────────────
    print("  Logging summary metrics...")
    experiment.log_metric("f1_range_across_conditions",         0.005)
    experiment.log_metric("anchor_score_range_across_conditions", 0.077)
    experiment.log_metric("best_ndcg_at_5",  PILOT_RESULTS["dense_only"]["ndcg_at_5"])
    experiment.log_metric("best_anchor_score", PILOT_RESULTS["hybrid_rerank"]["anchor_score"])
    experiment.log_metric("reranker_map_delta",   ABLATION_DELTAS["hybrid_to_hybrid_rerank"]["map_delta"])
    experiment.log_metric("reranker_anchor_delta", ABLATION_DELTAS["hybrid_to_hybrid_rerank"]["anchor_score_delta"])

    # ── 4. Statistical results ────────────────────────────────────────────────
    print("  Logging statistical test results...")
    experiment.log_metrics(STATISTICAL_RESULTS)

    # ── 5. Ablation deltas ────────────────────────────────────────────────────
    print("  Logging ablation delta table...")
    for step_name, deltas in ABLATION_DELTAS.items():
        for k, v in deltas.items():
            experiment.log_metric(f"ablation/{step_name}/{k}", v)

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print("  Uploading figures...")
    figures = {
        "fig1_metric_heatmap":          "Figure 1 — Full metric profile heatmap across 4 retrieval conditions",
        "fig2_anchor_score_comparison": "Figure 2 — AnchorScore vs Token F1 across conditions",
        "fig3_ablation_component":      "Figure 3 — Component ablation metric changes",
        "fig4_hallucination_rate":      "Figure 4 — Hallucination detection floor effect",
    }
    for fname, caption in figures.items():
        fpath = CHARTS_DIR / f"{fname}.png"
        if fpath.exists():
            experiment.log_image(str(fpath), name=fname, metadata={"caption": caption})
            print(f"    ✓ {fname}.png")
        else:
            print(f"    ⚠  {fpath} not found — skipping")

    # ── 7. Code artifact ─────────────────────────────────────────────────────
    print("  Creating code artifact...")
    try:
        artifact = Artifact(
            name="epraf-code-v1.0",
            artifact_type="code",
            metadata={
                "version":      "1.0",
                "github":       "https://github.com/reddy7356/EPRAF",
                "paper":        "EPRAF: A Proposed Hallucination-Aware Evaluation Framework "
                                "for Electrophysiology Anesthesia RAG",
                "submitted_to": "NEJM AI — Datasets, Benchmarks, and Protocols",
                "license":      "MIT",
            },
        )
        for py_file in sorted(CODE_DIR.glob("*.py")):
            artifact.add(str(py_file), logical_path=f"code/{py_file.name}")
        req_file = CODE_DIR / "requirements.txt"
        if req_file.exists():
            artifact.add(str(req_file), logical_path="code/requirements.txt")
        experiment.log_artifact(artifact)
        print("    ✓ Code artifact logged")
    except Exception as e:
        print(f"    ⚠  Code artifact skipped ({e})")

    # ── 8. Manuscript artifact ────────────────────────────────────────────────
    print("  Creating manuscript artifact...")
    try:
        doc_artifact = Artifact(
            name="epraf-manuscript-v1.0",
            artifact_type="docs",
            metadata={"format": "markdown", "target_journal": "NEJM AI"},
        )
        for doc_file in sorted(DOCS_DIR.iterdir()):
            doc_artifact.add(str(doc_file), logical_path=f"docs/{doc_file.name}")
        experiment.log_artifact(doc_artifact)
        print("    ✓ Manuscript artifact logged")
    except Exception as e:
        print(f"    ⚠  Manuscript artifact skipped ({e})")

    # ── 9. Done ───────────────────────────────────────────────────────────────
    experiment.end()

    url       = experiment.url or "(URL not available)"
    workspace = experiment.workspace or "(check Comet dashboard)"
    proj      = experiment.project_name or "epraf"

    print()
    print("=" * 60)
    print("✅  EPRAF experiment logged to Comet ML successfully!")
    print(f"🔗  Experiment URL : {url}")
    print(f"📁  Workspace      : {workspace}")
    print(f"📂  Project URL    : https://www.comet.com/{workspace}/{proj}")
    print()
    print("NEXT STEPS:")
    print("  1. Open the Experiment URL above — verify charts & metrics")
    print("  2. Make the project public:")
    print("     Comet UI → epraf project → ⚙ Settings → Visibility → Public")
    print("  3. Use the Project URL above in the manuscript Data Availability section")
    print("=" * 60)


if __name__ == "__main__":
    main()
