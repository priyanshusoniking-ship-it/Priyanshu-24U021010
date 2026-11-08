"""Evaluation benchmark comparing Transformer-based ranking against standard baselines.

Evaluates on:
- NDCG@5, NDCG@10
- MRR (Mean Reciprocal Rank)
- Hit Rate@5, Hit Rate@10
- Category Coverage & Shannon Entropy Diversity
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from models.ranker import NewsRanker
from models.transformer_encoder import NewsTransformerModel

BASE = Path(__file__).resolve().parent
DATA_CATALOG = BASE / "data" / "news_catalog.csv"
DATA_USERS = BASE / "data" / "user_profiles.json"
OUTPUT_DIR = BASE / "outputs"


def calculate_ndcg(ranked_ids: List[str], target_id: str, k: int) -> float:
    """Compute NDCG@K for a binary relevance target."""
    top_k = ranked_ids[:k]
    if target_id not in top_k:
        return 0.0
    rank = top_k.index(target_id) + 1
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(1 + 1)  # Ideal rank is 1
    return float(dcg / idcg)


def calculate_mrr(ranked_ids: List[str], target_id: str) -> float:
    """Compute Reciprocal Rank."""
    if target_id not in ranked_ids:
        return 0.0
    rank = ranked_ids.index(target_id) + 1
    return float(1.0 / rank)


def calculate_hit_rate(ranked_ids: List[str], target_id: str, k: int) -> float:
    """Binary hit rate at K."""
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def calculate_diversity(ranked_df: pd.DataFrame) -> float:
    """Shannon entropy over recommended categories."""
    if ranked_df.empty:
        return 0.0
    counts = ranked_df["category"].value_counts(normalize=True).values
    return float(-np.sum(counts * np.log2(counts + 1e-9)))


def evaluate_system() -> pd.DataFrame:
    catalog = pd.read_csv(DATA_CATALOG)
    with open(DATA_USERS, "r") as f:
        profiles = json.load(f)

    # Initialize and fit Transformer
    model = NewsTransformerModel(embed_dim=64, num_heads=4)
    model.fit_articles(catalog)
    ranker = NewsRanker(model)

    strategies = [
        ("Hybrid Transformer + Ranking", "hybrid_transformer"),
        ("Transformer-Only (Cosine)", "transformer_only"),
        ("Recency-Only (Chronological)", "recency_only"),
        ("Popularity-Only (Trending)", "popularity_only"),
    ]

    all_results = []

    for strat_label, strat_key in strategies:
        ndcg_5_list, ndcg_10_list = [], []
        mrr_list = []
        hit_5_list, hit_10_list = [], []
        diversity_list = []

        for p in profiles:
            history = p["history"]
            target = p["target_positive"]

            ranked_df, _ = ranker.rank_feed(
                catalog=catalog,
                history_ids=history,
                top_k=10,
                strategy=strat_key,
                apply_diversity=(strat_key == "hybrid_transformer"),
            )
            ranked_ids = ranked_df["news_id"].tolist()

            ndcg_5_list.append(calculate_ndcg(ranked_ids, target, k=5))
            ndcg_10_list.append(calculate_ndcg(ranked_ids, target, k=10))
            mrr_list.append(calculate_mrr(ranked_ids, target))
            hit_5_list.append(calculate_hit_rate(ranked_ids, target, k=5))
            hit_10_list.append(calculate_hit_rate(ranked_ids, target, k=10))
            diversity_list.append(calculate_diversity(ranked_df))

        all_results.append(
            {
                "Strategy": strat_label,
                "NDCG@5": np.mean(ndcg_5_list),
                "NDCG@10": np.mean(ndcg_10_list),
                "MRR": np.mean(mrr_list),
                "HitRate@5": np.mean(hit_5_list),
                "HitRate@10": np.mean(hit_10_list),
                "Diversity (Entropy)": np.mean(diversity_list),
            }
        )

    res_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

    print("\n" + "=" * 80)
    print("             PERSONALIZED NEWS FEED BENCHMARK RESULTS")
    print("=" * 80)
    print(res_df.round(4).to_string(index=False))
    print("=" * 80)
    print(f"Results successfully saved to {OUTPUT_DIR / 'evaluation_results.csv'}\n")

    return res_df


if __name__ == "__main__":
    evaluate_system()
