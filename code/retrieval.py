"""
Retrieval pipeline: sparse BM25, dense TF-IDF, hybrid RRF, cross-encoder reranking.
"""

import re
import math
import numpy as np
from collections import defaultdict
from config import Config
from data import BM25Index, TFIDFRetriever


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: list of lists of (doc_idx, score) pairs
        k: RRF constant

    Returns:
        Merged list of (doc_idx, rrf_score) sorted descending
    """
    rrf_scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked):
            rrf_scores[doc_idx] += 1.0 / (k + rank + 1)
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


class CrossEncoderReranker:
    """
    Deterministic cross-encoder reranker using token overlap scoring.
    Simulates cross-encoder behavior without external model dependency.
    """

    def __init__(self, min_overlap_tokens: int = 3):
        self.min_overlap_tokens = min_overlap_tokens

    def tokenize(self, text: str) -> set:
        return set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))

    def score(self, query: str, passage: str) -> float:
        """
        Score query-passage pair using:
        - Token overlap (Jaccard-like)
        - Length-normalized overlap
        - Bigram overlap bonus
        """
        q_tokens = self.tokenize(query)
        p_tokens = self.tokenize(passage)

        if not q_tokens or not p_tokens:
            return 0.0

        # Unigram overlap
        overlap = q_tokens & p_tokens
        precision = len(overlap) / len(q_tokens)
        recall = len(overlap) / len(p_tokens)

        # Always compute f1 unconditionally using ternary expression to
        # guarantee assignment regardless of the condition outcome.
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0
              else 0.0)
        # f1 is guaranteed assigned here

        # Bigram overlap bonus
        q_words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        p_words = re.findall(r'\b[a-zA-Z0-9]+\b', passage.lower())
        q_bigrams = set(zip(q_words, q_words[1:])) if len(q_words) > 1 else set()
        p_bigrams = set(zip(p_words, p_words[1:])) if len(p_words) > 1 else set()
        bigram_overlap = len(q_bigrams & p_bigrams)
        bigram_bonus = bigram_overlap / (len(q_bigrams) + 1)

        return f1 + 0.3 * bigram_bonus

    def rerank(self, query: str, candidates: list, corpus_texts: list,
               top_k: int = 5) -> list:
        """
        Rerank candidates by cross-encoder score.

        Args:
            query: query string
            candidates: list of (chunk_idx, score) pairs
            corpus_texts: list of chunk text strings
            top_k: number of results to return

        Returns:
            list of (chunk_idx, rerank_score) sorted descending
        """
        scored = []
        for chunk_idx, _ in candidates:
            text = corpus_texts[chunk_idx]
            s = self.score(query, text)
            scored.append((chunk_idx, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class SparseRetriever:
    """BM25-based sparse retrieval using exact keyword matching."""

    strategy_name = "bm25_sparse"

    def __init__(self, config: Config):
        self.config = config
        self.bm25 = BM25Index(k1=config.bm25_k1, b=config.bm25_b)
        self.fitted = False

    def fit(self, corpus_texts: list):
        self.bm25.fit(corpus_texts)
        self.fitted = True

    def retrieve(self, query: str, top_k: int = None) -> list:
        k = top_k if top_k is not None else self.config.top_k_sparse
        return self.bm25.retrieve(query, k)


class DenseRetriever:
    """TF-IDF dense-like retrieval using semantic vector similarity."""

    strategy_name = "tfidf_dense"

    def __init__(self, config: Config, max_features: int = 5000):
        self.config = config
        self.tfidf = TFIDFRetriever(max_features=max_features)
        self.fitted = False

    def fit(self, corpus_texts: list):
        self.tfidf.fit(corpus_texts)
        self.fitted = True

    def retrieve(self, query: str, top_k: int = None) -> list:
        k = top_k if top_k is not None else self.config.top_k_dense
        return self.tfidf.retrieve(query, k)


class HybridRetriever:
    """
    Hybrid retrieval combining sparse BM25 and dense TF-IDF via RRF,
    with optional cross-encoder reranking.
    """

    def __init__(self, config: Config, use_reranker: bool = True):
        self.config = config
        self.use_reranker = use_reranker
        self.sparse = SparseRetriever(config)
        self.dense = DenseRetriever(config)
        self.reranker = CrossEncoderReranker(
            min_overlap_tokens=config.min_citation_overlap_tokens
        )
        self.corpus_texts = []
        self.fitted = False

    def fit(self, corpus_texts: list):
        self.corpus_texts = corpus_texts
        self.sparse.fit(corpus_texts)
        self.dense.fit(corpus_texts)
        self.fitted = True

    def retrieve(self, query: str, top_k_final: int = None) -> list:
        # Always assign top_k_final before use — when None, fall back to config value.
        # This prevents UnboundLocalError if the None branch is not taken.
        top_k_final = None
        if top_k_final is None:
            top_k_final = self.config.top_k_rerank
        # top_k_final is guaranteed assigned here

        sparse_results = self.sparse.retrieve(query, self.config.top_k_sparse)
        dense_results = self.dense.retrieve(query, self.config.top_k_dense)

        # RRF fusion
        fused = reciprocal_rank_fusion(
            [sparse_results, dense_results],
            k=self.config.rrf_k
        )

        # Always assign result before returning — both branches assign it,
        # guaranteeing result is defined before the return statement.
        result = None
        if self.use_reranker:
            result = self.reranker.rerank(
                query, fused, self.corpus_texts, top_k=top_k_final
            )
        else:
            result = fused[:top_k_final]
        # result is guaranteed assigned here

        return result