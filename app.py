"""FeedTransformer AI — Personalized News Feed with Transformer Attention & Ranking.

Interactive Streamlit Web Dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models.ranker import NewsRanker
from models.transformer_encoder import NewsTransformerModel

BASE = Path(__file__).resolve().parent
DATA_CATALOG = BASE / "data" / "news_catalog.csv"
DATA_USERS = BASE / "data" / "user_profiles.json"
DATA_OUTPUTS = BASE / "outputs" / "evaluation_results.csv"

st.set_page_config(
    page_title="FeedTransformer AI | Personalized News Feed",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling & Design System
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #090e1a;
        --navy-card: #111a2e;
        --accent: #3b82f6;
        --accent-glow: #60a5fa;
        --cyan: #06b6d4;
        --emerald: #10b981;
        --amber: #f59e0b;
        --surface: #ffffff;
        --line: #e2e8f0;
        --ink: #0f172a;
        --muted: #64748b;
    }

    .stApp {
        background:
            radial-gradient(circle at 90% 10%, rgba(59,130,246,.08), transparent 28%),
            radial-gradient(circle at 10% 20%, rgba(6,182,212,.06), transparent 24%),
            #f8fafc;
    }

    .block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #0f172a 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stCaptionContainer p { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }

    .hero {
        position: relative; overflow: hidden; border-radius: 24px; padding: 2rem 2.2rem;
        background: linear-gradient(135deg, #090e1a 0%, #0f1c3f 50%, #1e3a8a 100%);
        box-shadow: 0 20px 60px rgba(15,23,42,.15); margin-bottom: 1.2rem; color: white;
    }
    .hero::after {
        content: ""; position: absolute; width: 340px; height: 340px; border-radius: 50%;
        right: -80px; top: -120px; background: radial-gradient(circle, rgba(96,165,250,.25), transparent 70%);
    }
    .hero-kicker { display: inline-flex; align-items: center; gap: .45rem; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #93c5fd; font-weight: 700; }
    .hero h1 { font-size: 2.45rem; margin: .35rem 0 .65rem; letter-spacing: -0.03em; color: white; }
    .hero p { color: #cbd5e1; font-size: .98rem; max-width: 880px; margin: 0; line-height: 1.5; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }
    .hero-badge { display: inline-flex; align-items: center; gap: .35rem; padding: .35rem .75rem; border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.15); font-size: .75rem; color: #e2e8f0; }

    .pill { display: inline-flex; align-items: center; gap: .3rem; border-radius: 999px; padding: .24rem .55rem; font-size: .72rem; font-weight: 700; }
    .pill-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #dbeafe; }
    .pill-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .pill-purple { background: #faf5ff; color: #7e22ce; border: 1px solid #f3e8ff; }
    .pill-amber { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .pill-slate { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

    .news-card {
        background: white; border: 1px solid var(--line); border-radius: 18px;
        padding: 1.1rem 1.2rem; margin-bottom: .85rem;
        box-shadow: 0 6px 20px rgba(15,23,42,.04); transition: all .15s ease;
    }
    .news-card:hover { border-color: #bfdbfe; box-shadow: 0 10px 28px rgba(15,23,42,.08); transform: translateY(-1px); }
    .card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: .7rem; }
    .rank-badge {
        width: 34px; height: 34px; border-radius: 10px; background: #eff6ff; color: #2563eb;
        display: flex; align-items: center; justify-content: center; font-weight: 850; font-size: .95rem;
    }
    .news-title { font-size: 1.05rem; font-weight: 750; color: #0f172a; margin-top: .15rem; line-height: 1.35; }
    .news-abstract { font-size: .84rem; color: #475569; margin-top: .45rem; line-height: 1.45; }
    .meta-row { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin-top: .65rem; }
    .score-bar-bg { height: 6px; border-radius: 999px; background: #f1f5f9; overflow: hidden; margin-top: .35rem; }
    .score-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #3b82f6, #06b6d4); }

    .stat-box {
        background: white; border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem; box-shadow: 0 6px 16px rgba(15,23,42,.03); text-align: center;
    }
    .stat-val { font-size: 1.6rem; font-weight: 850; color: #0f172a; }
    .stat-lbl { font-size: .74rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: .05em; }

    div[data-testid="stTabs"] { margin-top: .4rem; }
    button[data-baseweb="tab"] { font-weight: 750; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_and_fit_model():
    catalog = pd.read_csv(DATA_CATALOG)
    with open(DATA_USERS, "r") as f:
        profiles = json.load(f)
    model = NewsTransformerModel(embed_dim=64, num_heads=4)
    model.fit_articles(catalog)
    return catalog, profiles, model


catalog, profiles, transformer_model = load_and_fit_model()

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">⚡ NEURAL NLP + MULTI-HEAD ATTENTION + RE-RANKING</div>
        <h1>FeedTransformer AI</h1>
        <p>Personalized news feed generation combining a Transformer sequence encoder (Multi-Head Self-Attention over user reading history) with multi-factor ranking (semantic match, temporal freshness decay, category affinity, and diversity re-ranking).</p>
        <div class="hero-badges">
            <span class="hero-badge">✦ Multi-Head Self-Attention</span>
            <span class="hero-badge">✦ Additive Attention Pooling</span>
            <span class="hero-badge">✦ Exponential Freshness Decay</span>
            <span class="hero-badge">✦ Maximal Marginal Relevance (MMR)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Controls & Persona
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Select User Persona")
    persona_names = [f"{p['name']} ({p['persona']})" for p in profiles]
    selected_persona_idx = st.selectbox("Choose persona profile", range(len(persona_names)), format_func=lambda i: persona_names[i])
    current_persona = profiles[selected_persona_idx]

    # Initialize session state history if persona changed
    if "current_user_id" not in st.session_state or st.session_state["current_user_id"] != current_persona["user_id"]:
        st.session_state["current_user_id"] = current_persona["user_id"]
        st.session_state["user_history"] = list(current_persona["history"])

    st.caption(f"**Bio:** {current_persona['description']}")
    st.markdown("---")

    st.markdown("### ⚙️ Ranking Strategy")
    strategy_options = {
        "Hybrid Transformer + Ranking (Proposed)": "hybrid_transformer",
        "Transformer-Only (Cosine Semantic Match)": "transformer_only",
        "Chronological Freshness (Recency Baseline)": "recency_only",
        "Trending Clicks (Popularity Baseline)": "popularity_only",
    }
    chosen_strategy_label = st.radio("Active Model Strategy", list(strategy_options.keys()))
    strategy_key = strategy_options[chosen_strategy_label]

    st.markdown("---")
    st.markdown("### 🎛️ Ranking Hyperparameters")
    top_k = st.slider("Feed Length (Top-K)", 5, 15, 8)
    w_sim = st.slider("Semantic Transformer Weight", 0.0, 1.0, 0.55, 0.05)
    w_fresh = st.slider("Freshness Decay Weight", 0.0, 1.0, 0.25, 0.05)
    w_cat = st.slider("Category Affinity Weight", 0.0, 1.0, 0.15, 0.05)
    w_pop = 1.0 - (w_sim + w_fresh + w_cat)
    w_pop = max(0.0, round(w_pop, 2))
    st.caption(f"Popularity weight (remainder): **{w_pop:.2f}**")

    enable_diversity = st.checkbox("Apply MMR Diversity Re-Ranking", value=True)
    div_lambda = st.slider("MMR Relevance vs Diversity", 0.5, 1.0, 0.85, 0.05) if enable_diversity else 1.0

    if st.button("↻ Reset User History", use_container_width=True):
        st.session_state["user_history"] = list(current_persona["history"])
        st.rerun()

# -----------------------------------------------------------------------------
# Execute Ranking
# -----------------------------------------------------------------------------
ranker = NewsRanker(
    transformer_model=transformer_model,
    weight_similarity=w_sim,
    weight_freshness=w_fresh,
    weight_category=w_cat,
    weight_popularity=w_pop,
    diversity_lambda=div_lambda,
)

ranked_feed, diagnostics = ranker.rank_feed(
    catalog=catalog,
    history_ids=st.session_state["user_history"],
    top_k=top_k,
    strategy=strategy_key,
    apply_diversity=enable_diversity,
)

# -----------------------------------------------------------------------------
# KPI Overview Strip
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
history_len = len(st.session_state["user_history"])
avg_score = ranked_feed["ranking_score"].mean() if not ranked_feed.empty else 0.0
avg_fresh = ranked_feed["published_hours_ago"].mean() if not ranked_feed.empty else 0.0
cat_diversity = ranked_feed["category"].nunique() if not ranked_feed.empty else 0

with col1:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{history_len}</div><div class="stat-lbl">Articles Read</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(ranked_feed)}</div><div class="stat-lbl">Feed Size</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{avg_score:.3f}</div><div class="stat-lbl">Avg Rank Score</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{avg_fresh:.1f}h</div><div class="stat-lbl">Avg Freshness</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{cat_diversity}</div><div class="stat-lbl">Categories</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab_feed, tab_attn, tab_bench = st.tabs(["📰 Your Personalized Feed", "🧠 Transformer Attention & Insights", "📊 Benchmark & Evaluation"])

# --- TAB 1: FEED ---
with tab_feed:
    st.markdown(f"### 🎯 Recommended Feed for **{current_persona['name']}**")
    st.caption(f"Ranked using **{chosen_strategy_label}** • Interactive: Click 'Read Article' on any item to simulate reading it and dynamically update the feed!")

    if ranked_feed.empty:
        st.info("You have read all available news articles in the catalog!")
    else:
        for idx, row in ranked_feed.iterrows():
            pct = max(0, min(100, int(row["ranking_score"] * 100)))
            cat = row["category"]
            badge_class = "pill-blue" if "Tech" in cat else ("pill-green" if "Climate" in cat else ("pill-purple" if "Health" in cat else "pill-amber"))

            c_card, c_btn = st.columns([5.5, 1.2])
            with c_card:
                st.markdown(
                    f"""
                    <div class="news-card">
                        <div class="card-top">
                            <div class="rank-badge">#{int(row['rank'])}</div>
                            <div style="flex: 1;">
                                <span class="pill {badge_class}">{cat}</span>
                                <span class="pill pill-slate">{row['subcategory']}</span>
                                <div class="news-title">{row['title']}</div>
                                <div class="news-abstract">{row['abstract']}</div>
                                <div class="meta-row">
                                    <span class="pill pill-slate">🕒 {row['published_hours_ago']:.1f}h ago</span>
                                    <span class="pill pill-slate">👀 {int(row['popularity_score']):,} views</span>
                                    <span class="pill pill-slate">📖 {int(row['reading_time_min'])} min</span>
                                    <span class="pill pill-blue">Semantic: {row['sim_score']:.2f}</span>
                                    <span class="pill pill-green">Freshness: {row['fresh_score']:.2f}</span>
                                </div>
                                <div style="display:flex; justify-content:space-between; margin-top:.45rem; font-size:.74rem; color:#64748b;">
                                    <span>Total Neural Ranking Score</span>
                                    <b>{row['ranking_score']:.3f}</b>
                                </div>
                                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;"></div></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_btn:
                st.markdown('<div style="height: 38px;"></div>', unsafe_allow_html=True)
                if st.button("📖 Read", key=f"read_{row['news_id']}", use_container_width=True):
                    st.session_state["user_history"].append(row["news_id"])
                    st.rerun()

# --- TAB 2: TRANSFORMER ATTENTION & EXPLAINABILITY ---
with tab_attn:
    st.markdown("### 🧠 Multi-Head Attention & Representation Analysis")
    st.write(
        "The model uses **Multi-Head Self-Attention (MHSA)** over the user's reading sequence, followed by an **Additive Attention Pooling layer** to compute dynamic attention weights $\\alpha_i$ for each historical article."
    )

    col_l, col_r = st.columns([1.2, 1.0], gap="large")

    with col_l:
        st.markdown("#### 🎯 Reading History Attention Weights")
        hist_ids = diagnostics["history_ids"]
        pooling_weights = diagnostics["pooling_weights"]

        if len(hist_ids) > 0 and len(pooling_weights) > 0:
            hist_df = catalog[catalog["news_id"].isin(hist_ids)].set_index("news_id").loc[hist_ids].reset_index()
            hist_df["Attention Weight"] = pooling_weights
            hist_df["Short Title"] = hist_df["title"].apply(lambda t: t[:45] + "...")

            fig_attn = px.bar(
                hist_df,
                x="Attention Weight",
                y="Short Title",
                orientation="h",
                color="category",
                text_auto=".2%",
                title="Additive Attention Weight Distribution (User History)",
            )
            fig_attn.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                font=dict(color="#334155"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_attn, use_container_width=True)

            top_attended = hist_df.sort_values(by="Attention Weight", ascending=False).iloc[0]
            st.info(
                f"💡 **Explainability Insight:** The Transformer assigned the highest attention (**{top_attended['Attention Weight']:.1%}**) to **'{top_attended['title']}'** ({top_attended['category']}), driving current feed personalization."
            )
        else:
            st.write("No reading history recorded yet.")

    with col_r:
        st.markdown("#### 📊 Category Affinity & Distribution")
        cat_aff = diagnostics["category_affinity"]
        if cat_aff:
            aff_df = pd.DataFrame(list(cat_aff.items()), columns=["Category", "Affinity Share"])
            fig_pie = px.pie(
                aff_df,
                names="Category",
                values="Affinity Share",
                hole=0.45,
                title="Historical Reading Category Distribution",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 3: BENCHMARK & EVALUATION ---
with tab_bench:
    st.markdown("### 🏆 Benchmark Comparison Across Model Strategies")
    st.write(
        "Evaluation results across multiple test user personas demonstrating ranking accuracy (NDCG@K, MRR, HitRate) and topic diversity (Shannon Entropy)."
    )

    if DATA_OUTPUTS.exists():
        eval_df = pd.read_csv(DATA_OUTPUTS)
        st.dataframe(eval_df.style.highlight_max(subset=["NDCG@5", "NDCG@10", "MRR", "HitRate@5", "HitRate@10", "Diversity (Entropy)"], color="#dbeafe"), use_container_width=True)

        fig_comp = px.bar(
            eval_df,
            x="Strategy",
            y=["NDCG@5", "MRR", "HitRate@10"],
            barmode="group",
            text_auto=".2f",
            title="Ranking Metric Comparison (NDCG@5, MRR, HitRate@10)",
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Metric",
            xaxis_title="",
            font=dict(color="#334155"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Run `python evaluate.py` to generate the latest benchmark table.")

    st.markdown("---")
    st.markdown("### 📐 Mathematical Formulation")
    st.latex(r"Q = X W_Q, \quad K = X W_K, \quad V = X W_V")
    st.latex(r"\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V")
    st.latex(r"\text{Score}(j) = w_{\text{sim}} \cdot \text{sim}(\mathbf{u}, \mathbf{a}_j) + w_{\text{fresh}} \cdot e^{-\frac{\Delta t}{48}} + w_{\text{cat}} \cdot \text{affinity} + w_{\text{pop}} \cdot \text{popularity}")
