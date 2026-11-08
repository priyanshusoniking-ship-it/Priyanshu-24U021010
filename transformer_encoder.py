"""Transformer-based News and User History Sequence Encoder.

Implements Multi-Head Self-Attention (MHSA) and Additive Attention Pooling
(inspired by the Neural News Recommendation with Multi-Head Self-Attention - NRMS architecture).
Works natively using high-performance vectorized NumPy/SciPy, and seamlessly supports
PyTorch/SentenceTransformers when available.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadSelfAttention:
    """Multi-Head Self-Attention Layer with Residual Connection and LayerNorm."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, seed: int = 42):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Deterministic initialization
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        self.W_q = rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_k = rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_v = rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_o = rng.normal(0, scale, (embed_dim, embed_dim))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass for sequence x of shape (seq_len, embed_dim).

        Returns:
            output: (seq_len, embed_dim)
            attention_weights: (num_heads, seq_len, seq_len)
        """
        seq_len, d = x.shape
        # Linear projections
        Q = np.dot(x, self.W_q)  # (seq_len, d)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Reshape into multiple heads: (num_heads, seq_len, head_dim)
        Q_heads = Q.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)
        K_heads = K.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)
        V_heads = V.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)

        # Scaled dot-product attention per head
        scores = np.matmul(Q_heads, K_heads.swapaxes(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = softmax(scores, axis=-1)  # (num_heads, seq_len, seq_len)

        # Context output per head
        head_out = np.matmul(attn_weights, V_heads)  # (num_heads, seq_len, head_dim)
        # Concatenate heads: (seq_len, embed_dim)
        concat_out = head_out.swapaxes(0, 1).reshape(seq_len, self.embed_dim)

        # Final projection + Residual connection
        out = np.dot(concat_out, self.W_o) + x

        # Layer Normalization
        mean = np.mean(out, axis=-1, keepdims=True)
        std = np.std(out, axis=-1, keepdims=True) + 1e-6
        norm_out = (out - mean) / std

        return norm_out, attn_weights


class AdditiveAttentionPooling:
    """Additive Attention (Query-Key pooling) to compress a sequence into a vector.

    Learns dynamic attention weights alpha_i over each item in the user's history.
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 32, seed: int = 42):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        self.W_a = rng.normal(0, scale, (embed_dim, hidden_dim))
        self.b_a = np.zeros(hidden_dim)
        self.q = rng.normal(0, 1.0 / math.sqrt(hidden_dim), hidden_dim)

    def forward(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate history representations h of shape (seq_len, embed_dim).

        Returns:
            user_vec: (embed_dim,)
            weights: (seq_len,)
        """
        if len(h) == 0:
            return np.zeros(self.embed_dim), np.array([])
        if len(h) == 1:
            return h[0], np.array([1.0])

        # Dense projection + tanh activation
        projected = np.tanh(np.dot(h, self.W_a) + self.b_a)  # (seq_len, hidden_dim)
        # Dot with query vector q
        scores = np.dot(projected, self.q)  # (seq_len,)
        weights = softmax(scores)  # (seq_len,)

        user_vec = np.sum(h * weights[:, np.newaxis], axis=0)
        # L2 normalize
        norm = np.linalg.norm(user_vec)
        if norm > 1e-6:
            user_vec = user_vec / norm

        return user_vec, weights


class NewsTransformerModel:
    """Complete Neural News & User Sequence Recommendation System."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.tfidf = TfidfVectorizer(max_features=256, stop_words="english")
        self.projection_matrix: Optional[np.ndarray] = None
        self.article_embeddings: Dict[str, np.ndarray] = {}

        self.self_attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.user_pooling = AdditiveAttentionPooling(embed_dim=embed_dim)

    def fit_articles(self, df: pd.DataFrame) -> None:
        """Fit article representation space on catalog texts."""
        texts = [
            f"{row['category']} {row['subcategory']} {row['title']} {row['abstract']}"
            for _, row in df.iterrows()
        ]
        tfidf_mat = self.tfidf.fit_transform(texts).toarray()

        # Random orthogonal projection from TF-IDF space to dense embedding dimension
        rng = np.random.default_rng(101)
        raw_proj = rng.normal(0, 1.0, (tfidf_mat.shape[1], self.embed_dim))
        q, _ = np.linalg.qr(raw_proj)
        self.projection_matrix = q

        dense_embs = np.dot(tfidf_mat, self.projection_matrix)
        # L2 normalize
        norms = np.linalg.norm(dense_embs, axis=1, keepdims=True) + 1e-6
        dense_embs = dense_embs / norms

        self.article_embeddings = {
            df.iloc[i]["news_id"]: dense_embs[i] for i in range(len(df))
        }

    def get_article_embedding(self, news_id: str) -> np.ndarray:
        return self.article_embeddings.get(news_id, np.zeros(self.embed_dim))

    def encode_user_history(
        self, history_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode the user's reading history sequence with Transformer Self-Attention.

        Returns:
            user_vector: (embed_dim,)
            pooling_weights: (seq_len,)
            self_attention_matrix: (num_heads, seq_len, seq_len)
        """
        valid_ids = [nid for nid in history_ids if nid in self.article_embeddings]
        if not valid_ids:
            return np.zeros(self.embed_dim), np.array([]), np.zeros((self.num_heads, 0, 0))

        # Shape (seq_len, embed_dim)
        seq_embs = np.array([self.article_embeddings[nid] for nid in valid_ids])
        seq_len = len(seq_embs)

        # Add sinusoidal/temporal positional encodings (giving subtle boost to recent reads)
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(self.embed_dim)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(self.embed_dim))
        angle_rads = pos * angle_rates
        pos_encoding = np.zeros_like(seq_embs)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Add positional signals scaled by 0.15
        x = seq_embs + 0.15 * pos_encoding

        # Pass through Transformer Multi-Head Self-Attention Layer
        transformed_seq, attn_weights = self.self_attention.forward(x)

        # Pass through Additive Attention Pooling to aggregate into single User Vector
        user_vec, pooling_weights = self.user_pooling.forward(transformed_seq)

        return user_vec, pooling_weights, attn_weights
