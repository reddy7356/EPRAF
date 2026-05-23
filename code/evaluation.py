"""
Evaluation metrics: retrieval (precision, recall, MRR, NDCG),
QA accuracy (EM, F1), hallucination rates, and statistical tests.
"""

import re
import math
import numpy as np
from collections import defaultdict
from config import Config


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    top_k = [idx for idx, _ in retrieved[:k]]
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Recall@K: fraction of relevant docs retrieved in top-K."""
    if not relevant:
        return 0.0
    top_k = [idx for idx, _ in retrieved[:k]]
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list, relevant: set) -> float:
    """MRR: reciprocal of rank of first relevant document."""
    for rank, (idx, _) in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    top_k = [idx for idx, _ in retrieved[:k]]
    dcg = 0.0
    for i, idx in enumerate(top_k):
        if idx in relevant:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: all relevant docs at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: list, relevant: set) -> float:
    """Average Precision (AP) for a single query."""
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, (idx, _) in enumerate(retrieved, start=1):
        if idx in relevant:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant)


def compute_retrieval_metrics(
    results_per_query: list, k_values: list = None
) -> dict:
    """
    Compute aggregate retrieval metrics across queries.

    Args:
        results_per_query: list of dicts with keys 'retrieved', 'relevant'
        k_values: list of K values for P@K, R@K, NDCG@K

    Returns:
        dict of metric_name -> float
    """
    k_values = None
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = defaultdict(list)

    for entry in results_per_query:
        retrieved = entry["retrieved"]  # list of (idx, score)
        relevant = entry["relevant"]  # set of relevant idx

        for k in k_values:
            metrics[f"P@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"R@{k}"].append(recall_at_k(retrieved, relevant, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(retrieved, relevant, k))

        metrics["MRR"].append(mean_reciprocal_rank(retrieved, relevant))
        metrics["AP"].append(average_precision(retrieved, relevant))

    # Average across queries
    aggregated = {k: float(np.mean(v)) for k, v in metrics.items()}
    aggregated["MAP"] = aggregated.pop("AP", 0.0)
    return aggregated


# ---------------------------------------------------------------------------
# QA accuracy metrics (Exact Match and Token F1)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, remove punctuation/articles."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    pred_counter = defaultdict(int)
    for t in pred_tokens:
        pred_counter[t] += 1
    gt_counter = defaultdict(int)
    for t in gt_tokens:
        gt_counter[t] += 1

    common = 0
    for t in gt_counter:
        common += min(gt_counter[t], pred_counter.get(t, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_qa_metrics(predictions: list, ground_truths: list) -> dict:
    """
    Compute EM and F1 across prediction-ground_truth pairs.

    Args:
        predictions: list of predicted answer strings
        ground_truths: list of ground truth answer strings

    Returns:
        dict with 'exact_match', 'f1', 'n_samples'
    """
    assert len(predictions) == len(ground_truths), "Length mismatch"
    ems = []
    f1s = []
    for pred, gt in zip(predictions, ground_truths):
        ems.append(exact_match(pred, gt))
        f1s.append(token_f1(pred, gt))
    return {
        "exact_match": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "n_samples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def mcnemar_test(
    correct_a: list, correct_b: list, continuity_correction: bool = True
) -> dict:
    """
    McNemar's test for paired binary outcomes.

    Args:
        correct_a: list of 0/1 correctness for system A
        correct_b: list of 0/1 correctness for system B
        continuity_correction: apply Yates continuity correction

    Returns:
        dict with 'chi2', 'p_value', 'b', 'c', 'significant'
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0 and bb == 1)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    chi2 = None
    if continuity_correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)

    p_value = _chi2_sf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "b": b,
        "c": c,
        "significant": p_value < 0.05,
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """
    Survival function of chi-squared distribution (1 - CDF).
    Uses regularized incomplete gamma function approximation for df=1.
    """
    if x <= 0:
        return 1.0
    # For df=1: chi2_sf(x) = erfc(sqrt(x/2))
    return _erfc(math.sqrt(x / 2))


def _erfc(x: float) -> float:
    """
    Complementary error function approximation (Abramowitz & Stegun).

    The polynomial approximation is valid only for x >= 0.
    For x < 0 we use the identity erfc(-x) = 2 - erfc(x), applying
    the polynomial to the positive argument and then reflecting.
    Maximum absolute error of the polynomial for x >= 0 is ~1.5e-7.
    """
    if x < 0:
        # Reflect: erfc(-x) = 2 - erfc(x)
        return 2.0 - _erfc(-x)
    # Polynomial valid for x >= 0
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (
            -0.284496736
            + t * (
                1.421413741
                + t * (-1.453152027 + t * 1.061405429)
            )
        )
    )
    return poly * math.exp(-x * x)


def wilcoxon_signed_rank_test(scores_a: list, scores_b: list) -> dict:
    """
    Wilcoxon signed-rank test for paired continuous scores.

    Uses tie-corrected variance:
        Var(W) = [n(n+1)(2n+1)/24] - [sum(t_j^3 - t_j)/48]
    where t_j is the size of each tie group.

    Rank assignment: for a group of tied absolute differences spanning
    0-based positions i..j-1, the correct 1-based average rank is
    (i + 1 + j) / 2  (e.g., positions 0,1 -> ranks 1,2 -> avg 1.5).

    Returns:
        dict with 'W', 'p_value', 'significant'
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    nonzero_diffs = [(i, d) for i, d in enumerate(diffs) if d != 0]

    if len(nonzero_diffs) == 0:
        return {"W": 0.0, "p_value": 1.0, "significant": False}

    # Rank absolute differences
    abs_diffs = [(i, abs(d), d) for i, d in nonzero_diffs]
    abs_diffs.sort(key=lambda tup: tup[1])

    # Assign ranks with ties averaged; also collect tie group sizes.
    # For a tie group spanning 0-based positions i..j-1 (inclusive),
    # the 1-based ranks are (i+1)..(j), so the average 1-based rank is
    # (i + 1 + j) / 2.
    ranks = []
    tie_group_sizes = []
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][1] == abs_diffs[i][1]:
            j += 1
        group_size = j - i
        tie_group_sizes.append(group_size)
        # Correct 1-based average rank for positions i..j-1 (0-based):
        # lowest 1-based rank = i+1, highest = j, average = (i+1+j)/2
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks.append((abs_diffs[k][0], avg_rank, abs_diffs[k][2]))
        i = j

    W_plus = sum(r for _, r, d in ranks if d > 0)
    W_minus = sum(r for _, r, d in ranks if d < 0)
    W = min(W_plus, W_minus)

    n = len(ranks)
    mean_W = n * (n + 1) / 4

    # Tie-corrected variance: subtract sum(t_j^3 - t_j)/48
    base_var = n * (n + 1) * (2 * n + 1) / 24
    tie_correction = sum(t ** 3 - t for t in tie_group_sizes) / 48
    var_W = base_var - tie_correction

    if var_W <= 0:
        return {"W": float(W), "p_value": 1.0, "significant": False}

    std_W = math.sqrt(var_W)
    z = (W - mean_W) / std_W
    p_value = 2 * _normal_sf(abs(z))  # two-tailed

    return {
        "W": float(W),
        "z": float(z),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def _normal_sf(z: float) -> float:
    """Standard normal survival function P(Z > z). Valid for all real z."""
    # _erfc handles both positive and negative arguments correctly
    return 0.5 * _erfc(z / math.sqrt(2))


def compute_power_analysis(
    effect_size: float, n: int, alpha: float = 0.05
) -> float:
    """
    Compute statistical power for a two-proportion z-test.

    Args:
        effect_size: Cohen's h effect size
        n: sample size per group
        alpha: significance level

    Returns:
        power estimate in [0, 1]
    """
    z_alpha = _normal_quantile(1 - alpha / 2)
    lambda_nc = effect_size * math.sqrt(n)
    power = _normal_sf(z_alpha - lambda_nc) + (
        1 - _normal_sf(-z_alpha - lambda_nc)
    )
    return min(1.0, max(0.0, power))


def _normal_quantile(p: float) -> float:
    """
    Inverse normal CDF approximation (Beasley-Springer-Moro algorithm).
    """
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')

    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511349,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ]

    y = p - 0.5
    r = None
    x = None
    if abs(y) < 0.42:
        r = y * y
        x = (
            y
            * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
            / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)
        )
    else:
        r = p if y < 0 else 1 - p
        r = math.log(-math.log(r))
        x = c[0] + r * (
            c[1]
            + r * (
                c[2]
                + r * (
                    c[3]
                    + r * (
                        c[4]
                        + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))
                    )
                )
            )
        )
        if y < 0:
            x = -x
    return x


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(max(0, min(1, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0, min(1, p2))))
    return abs(phi1 - phi2)


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """
    Compute Cohen's kappa for inter-rater reliability between two binary raters.

    Args:
        labels_a: list of binary labels (0/1) from rater A
        labels_b: list of binary labels (0/1) from rater B

    Returns:
        Cohen's kappa in [-1, 1]
    """
    assert len(labels_a) == len(labels_b), "Label lists must have equal length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement by chance
    p_a1 = sum(labels_a) / n  # P(rater A says 1)
    p_b1 = sum(labels_b) / n  # P(rater B says 1)
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if abs(1 - p_e) < 1e-12:
        return 1.0  # Perfect agreement by chance as well

    return (p_o - p_e) / (1 - p_e)