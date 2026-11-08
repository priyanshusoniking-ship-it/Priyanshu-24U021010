"""Multi-Factor Neural Ranking and Diversity Re-Ranking Engine.

Combines Transformer Semantic Similarity with Temporal Freshness Decay,
Category Affinity, Popularity Signals, and Maximal Marginal Relevance (MMR)
to create a balanced, engaging, personalized news feed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.transformer_encoder import NewsTransformerModel


class NewsRanker:
    """Multi-factor news ranker and diversity engine."""

    def __init__(
        self,
        transformer_model: NewsTransformerModel,
        weight_similarity: float = 0.55,
        weight_freshness: float = 0.25,
        weight_category: float = 0.15,
        weight_popularity: float = 0.05,
        diversity_lambda: float = 0.85,
    ):
        self.model = transformer_model
        self.w_sim = weight_similarity
        self.w_fresh = weight_freshness
        self.w_cat = weight_category
        self.w_pop = weight_popularity
        self.diversity_lambda = diversity_lambda

    def rank_feed(
        self,
        catalog: pd.DataFrame,
        history_ids: List[str],
        top_k: int = 10,
        strategy: str = "hybrid_transformer",
        apply_diversity: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Rank unread candidate news articles for a user.

        Strategies:
            - "hybrid_transformer": Transformer similarity + Freshness + Category + Popularity
            - "transformer_only": Pure semantic cosine similarity
            - "recency_only": Chronological sorting by freshness
            - "popularity_only": Trending sort by view counts
        """
        # Filter out already read articles
        candidates = catalog[~catalog["news_id"].isin(history_ids)].copy().reset_index(drop=True)
        if candidates.empty:
            return pd.DataFrame(), {}

        # 1. User Transformer Representation
        user_vec, pooling_weights, attn_matrix = self.model.encode_user_history(history_ids)

        # 2. Extract Category Affinity from History
        read_articles = catalog[catalog["news_id"].isin(history_ids)]
        cat_counts = read_articles["category"].value_counts(normalize=True).to_dict()

        # Maximum popularity for normalization
        max_pop = catalog["popularity_score"].max() if not catalog.empty else 1.0

        scores_list = []
        sim_scores = []
        fresh_scores = []
        pop_scores = []
        cat_scores = []

        for _, row in candidates.iterrows():
            nid = row["news_id"]
            emb = self.model.get_article_embedding(nid)

            # Cosine semantic similarity: [-1, 1] scaled to [0, 1]
            sim = float(np.dot(user_vec, emb)) if np.linalg.norm(user_vec) > 1e-6 else 0.0
            sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))

            # Exponential decay for freshness: exp(-hours / 48)
            hours = float(row.get("published_hours_ago", 24.0))
            freshness = float(np.exp(-hours / 48.0))

            # Popularity normalized with log scaling
            pop = float(row.get("popularity_score", 100))
            pop_norm = float(np.log1p(pop) / np.log1p(max_pop))

            # Category affinity
            cat = str(row["category"])
            cat_affinity = float(cat_counts.get(cat, 0.0))

            sim_scores.append(sim_norm)
            fresh_scores.append(freshness)
            pop_scores.append(pop_norm)
            cat_scores.append(cat_affinity)

            # Strategy-based scoring
            if strategy == "hybrid_transformer":
                final_score = (
                    self.w_sim * sim_norm
                    + self.w_fresh * freshness
                    + self.w_cat * cat_affinity
                    + self.w_pop * pop_norm
                )
            elif strategy == "transformer_only":
                final_score = sim_norm
            elif strategy == "recency_only":
                final_score = freshness
            elif strategy == "popularity_only":
                final_score = pop_norm
            else:
                final_score = sim_norm

            scores_list.append(final_score)

        candidates["ranking_score"] = scores_list
        candidates["sim_score"] = sim_scores
        candidates["fresh_score"] = fresh_scores
        candidates["pop_score"] = pop_scores
        candidates["cat_affinity"] = cat_scores

        # 3. Diversity / Re-ranking (MMR)
        if apply_diversity and strategy == "hybrid_transformer" and len(candidates) > top_k:
            ranked_df = self._maximal_marginal_relevance(candidates, top_k=top_k)
        else:
            ranked_df = candidates.sort_values(by="ranking_score", ascending=False).head(top_k).reset_index(drop=True)

        ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)

        diagnostics = {
            "user_vector": user_vec,
            "pooling_weights": pooling_weights,
            "attn_matrix": attn_matrix,
            "category_affinity": cat_counts,
            "history_ids": history_ids,
        }
        return ranked_df, diagnostics

    def _maximal_marginal_relevance(self, candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """MMR algorithm to balance relevance score with inter-article diversity."""
        pool = candidates.copy().reset_index(drop=True)
        selected_indices: List[int] = []

        # Start with the highest scoring item
        best_idx = int(pool["ranking_score"].idxmax())
        selected_indices.append(best_idx)

        # Iteratively pick items that maximize MMR
        for _ in range(1, min(top_k, len(pool))):
            best_mmr = -1e9
            best_cand_idx = -1

            for idx, row in pool.iterrows():
                if idx in selected_indices:
                    continue

                curr_emb = self.model.get_article_embedding(row["news_id"])
                # Compute maximum similarity to already selected articles
                max_sim_to_selected = max(
                    float(np.dot(curr_emb, self.model.get_article_embedding(pool.loc[s, "news_id"])))
                    for s in selected_indices
                )
                max_sim_norm = max(0.0, min(1.0, (max_sim_to_selected + 1.0) / 2.0))

                relevance = float(row["ranking_score"])
                mmr_score = (self.diversity_lambda * relevance) - ((1.0 - self.diversity_lambda) * max_sim_norm)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cand_idx = idx

            if best_cand_idx != -1:
                selected_indices.append(best_cand_idx)

        return pool.loc[selected_indices].reset_index(drop=True)
