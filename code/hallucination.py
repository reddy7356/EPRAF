"""
Hallucination detection via evidence anchoring and citation overlap analysis.

Overlap is measured as a NORMALIZED fraction (shared tokens / claim tokens)
to avoid the raw-count bias that favours long sentences and common stopwords.
Medical stopwords are filtered before comparison so that common function words
cannot trivially satisfy the support threshold.
"""

import re
import numpy as np
from config import Config


# ---------------------------------------------------------------------------
# Medical stopwords to exclude from overlap computation
# ---------------------------------------------------------------------------

MEDICAL_STOPWORDS = {
    "the", "is", "of", "and", "in", "a", "an", "to", "for", "or",
    "with", "that", "this", "it", "be", "are", "was", "were", "as",
    "at", "by", "from", "on", "not", "may", "can", "has", "have",
    "been", "its", "which", "such", "than", "more", "also", "used",
    "use", "using", "during", "after", "before", "when", "if", "should",
    "must", "will", "would", "could", "their", "they", "we", "he",
    "she", "his", "her", "our", "your", "up", "down", "into", "out",
    "about", "between", "through", "over", "under", "per", "both",
    "each", "all", "any", "other", "following", "required", "needed",
}


def tokenize_simple(text: str) -> list:
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def tokenize_content(text: str) -> set:
    """Return content tokens (stopwords removed) as a set."""
    return {
        t for t in tokenize_simple(text)
        if t not in MEDICAL_STOPWORDS and len(t) > 1
    }


def compute_normalized_overlap(claim: str, evidence: str) -> float:
    """
    Compute normalized overlap between claim and evidence.

    Returns the fraction of *claim* content tokens that appear in the
    evidence content tokens (i.e., claim-side recall).  This is
    length-normalized so that short claims are not trivially penalized
    and long verbose sentences are not trivially rewarded.
    """
    claim_tokens = tokenize_content(claim)
    evidence_tokens = tokenize_content(evidence)

    if not claim_tokens:
        return 0.0

    shared = claim_tokens & evidence_tokens
    return len(shared) / len(claim_tokens)


def extract_claims(answer: str) -> list:
    """
    Split answer into atomic claims (sentences or clauses).
    Returns list of claim strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 10]
    return claims


def check_claim_against_evidence(
    claim: str,
    evidence_chunks: list,
    min_overlap_fraction: float = 0.25,
) -> dict:
    """
    Check whether a claim is supported by at least one evidence chunk.

    Support is determined by normalized overlap (fraction of claim content
    tokens found in the evidence chunk), NOT raw token count.

    Args:
        claim: claim string to verify
        evidence_chunks: list of evidence chunk text strings
        min_overlap_fraction: minimum fraction of claim content tokens that
            must appear in an evidence chunk to count as supported

    Returns:
        dict with keys:
            - supported: bool
            - best_overlap: float (normalized, 0-1)
            - best_chunk_idx: int or None
    """
    best_overlap = 0.0
    best_chunk_idx = None

    for i, chunk_text in enumerate(evidence_chunks):
        overlap = compute_normalized_overlap(claim, chunk_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk_idx = i

    supported = best_overlap >= min_overlap_fraction
    return {
        "supported": supported,
        "best_overlap": best_overlap,
        "best_chunk_idx": best_chunk_idx,
    }


def detect_hallucinations(
    answer: str, evidence_chunks: list, config: Config
) -> dict:
    """
    Detect hallucinated claims in an answer given retrieved evidence.

    Algorithm:
    1. Extract atomic claims from answer
    2. For each claim, compute normalized content-token overlap with evidence
    3. Claims with overlap < min_citation_overlap_fraction are flagged
    4. Hallucination score = fraction of unsupported claims
    5. Answer is flagged if unsupported_count >= hallucination_flag_threshold

    Returns:
        dict with:
            - claims: list of claim strings
            - claim_results: list of check results
            - n_claims: int
            - n_unsupported: int
            - hallucination_score: float [0, 1]
            - is_hallucinated: bool
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "claims": [],
            "claim_results": [],
            "n_claims": 0,
            "n_unsupported": 0,
            "hallucination_score": 0.0,
            "is_hallucinated": False,
        }

    claim_results = []
    n_unsupported = 0

    for claim in claims:
        result = check_claim_against_evidence(
            claim,
            evidence_chunks,
            min_overlap_fraction=config.min_citation_overlap_fraction,
        )
        claim_results.append(result)
        if not result["supported"]:
            n_unsupported += 1

    hallucination_score = n_unsupported / len(claims)
    is_hallucinated = n_unsupported >= config.hallucination_flag_threshold

    return {
        "claims": claims,
        "claim_results": claim_results,
        "n_claims": len(claims),
        "n_unsupported": n_unsupported,
        "hallucination_score": hallucination_score,
        "is_hallucinated": is_hallucinated,
    }


def compute_evidence_anchor_score(
    answer: str, evidence_chunks: list, scale: int = 5
) -> float:
    """
    Compute evidence anchor score on [0, scale].

    Uses normalized overlap (fraction of answer content tokens found in
    the best-matching evidence chunk) to avoid length bias.
    """
    if not evidence_chunks or not answer:
        return 0.0

    answer_tokens = tokenize_content(answer)
    if not answer_tokens:
        return 0.0

    max_overlap = 0.0
    for chunk in evidence_chunks:
        overlap = compute_normalized_overlap(answer, chunk)
        if overlap > max_overlap:
            max_overlap = overlap

    return round(max_overlap * scale, 3)


def generate_answer_from_evidence(
    query: str, evidence_chunks: list, compression_ratio: float = 0.5
) -> str:
    """
    Generate a deterministic answer by extracting relevant sentences
    from evidence chunks (extractive QA approach).

    Sentences are scored by NORMALIZED query-token overlap (overlap count
    divided by sentence content-token count) to avoid the length bias that
    favours verbose sentences with more total tokens.

    Args:
        query: query string
        evidence_chunks: list of relevant chunk texts
        compression_ratio: fraction of candidate sentences to include

    Returns:
        Extracted answer string
    """
    if not evidence_chunks:
        return "No relevant evidence found."

    query_tokens = tokenize_content(query)

    # Score each sentence by normalized overlap with query
    all_sentences = []
    for chunk in evidence_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15:
                sent_tokens = tokenize_content(sent)
                if sent_tokens:
                    # Normalize by sentence content-token count to avoid
                    # systematic bias toward longer sentences
                    overlap_count = len(query_tokens & sent_tokens)
                    normalized_score = overlap_count / len(sent_tokens)
                else:
                    normalized_score = 0.0
                all_sentences.append((sent, normalized_score))

    if not all_sentences:
        return evidence_chunks[0][:200]

    # Sort by normalized overlap descending
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    # Take top fraction
    n_keep = max(1, int(len(all_sentences) * compression_ratio))
    selected = [s for s, _ in all_sentences[:n_keep]]

    return " ".join(selected)