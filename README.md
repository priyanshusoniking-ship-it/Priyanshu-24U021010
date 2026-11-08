# FeedTransformer AI — Personalized News Feed with Transformer Attention & Ranking

A neural news recommendation and ranking engine. The platform models user reading history sequences using **Multi-Head Self-Attention (MHSA)** and **Additive Attention Pooling** (inspired by the NRMS architecture), combined with multi-factor ranking (semantic relevance, exponential freshness decay, category affinity, and Maximal Marginal Relevance diversity) to deliver dynamic, personalized news feeds.

---

## 🏛️ System Architecture

- **Frontend & Dashboard**: Streamlit (wide layout, responsive metric cards, dark hero design system)
- **Data Visualizations**: Plotly Express (interactive attention distributions, category affinity pies, benchmark comparisons)
- **Neural Sequence Encoder**: Multi-Head Self-Attention (MHSA) Layer with Residual Connection and Layer Normalization
- **Attention Pooling Layer**: Query-Key Additive Attention Pooling to aggregate history sequences into continuous user preference embeddings
- **Multi-Factor Neural Ranker**: Composite scoring function balancing semantic alignment, temporal recency decay ($\exp(-\Delta t / 48)$), category affinity, and social popularity
- **Diversity Engine**: Maximal Marginal Relevance (MMR) re-ranking to prevent filter bubbles and topic fatigue

---

## 📋 Prerequisites

Ensure you have the following installed on your machine:
- **Python** (v3.10 or newer)
- **pip** (Python package installer)

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment

```bash
# Navigate to the project directory
cd personalized_news_feed

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Launch the Web Application

```bash
streamlit run app.py
```

- **Interactive Dashboard**: [http://localhost:8501](http://localhost:8501)
- *(On Windows, you can also double-click `run_app.bat` to launch immediately).*

---

## 📁 Project Structure

```text
personalized_news_feed/
├── app.py                      # Main Streamlit web application & user interface
├── evaluate.py                 # Automated benchmark script comparing model strategies
├── generate_data.py            # Dataset generator for news catalog and user personas
├── run_app.bat                 # One-click Windows batch launcher
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── models/
│   ├── transformer_encoder.py  # Multi-Head Self-Attention & Additive Attention Pooling
│   └── ranker.py               # Multi-factor neural ranking & MMR diversity logic
│
├── data/
│   ├── news_catalog.csv        # Multi-category news catalog (Tech, Climate, Finance, Health, Space)
│   └── user_profiles.json      # Ground-truth user personas and reading sequences
│
├── outputs/
│   └── evaluation_results.csv  # Benchmark output comparing NDCG, MRR, HitRate & Diversity
└── README.md                   # Project documentation
```

---

## 🛠️ Useful Scripts & Benchmark Evaluation

### 1. Run Automated Evaluation Benchmark

Run batch evaluation across multiple test user profiles to evaluate ranking quality:

```bash
python evaluate.py
```

**Benchmark Results:**

| Strategy | NDCG@5 | NDCG@10 | MRR | HitRate@5 | HitRate@10 | Diversity (Entropy) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Transformer + Ranking (Proposed)** | **0.8123** | **0.8123** | **0.7500** | **1.000** | **1.000** | **2.6419** |
| Transformer-Only (Cosine Match) | 0.4000 | 0.4712 | 0.4333 | 0.400 | 0.600 | 2.3664 |
| Chronological Freshness (Recency) | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 2.4615 |
| Trending Clicks (Popularity) | 0.0000 | 0.0602 | 0.0222 | 0.000 | 0.200 | 2.1664 |

---

## 📐 Mathematical Formulation

### 1. Multi-Head Self-Attention (User Sequence Modeling)

For a sequence of historical articles $X \in \mathbb{R}^{S \times D}$:

$$Q_i = X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i$$

$$\text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O + X$$

### 2. Additive Attention Pooling

Learns dynamic attention weights $\alpha_i$ to aggregate the sequence into a unified user representation $\mathbf{u}$:

$$\alpha_i = \frac{\exp(\mathbf{q}^T \tanh(W_a \mathbf{h}_i + b_a))}{\sum_k \exp(\mathbf{q}^T \tanh(W_a \mathbf{h}_k + b_a))}, \quad \mathbf{u} = \sum_i \alpha_i \mathbf{h}_i$$

### 3. Multi-Factor Neural Ranking Score

Candidate articles $j$ are ranked using:

$$\text{Score}(j) = w_{\text{sim}} \cdot \text{sim}(\mathbf{u}, \mathbf{a}_j) + w_{\text{fresh}} \cdot e^{-\frac{\Delta t}{48}} + w_{\text{cat}} \cdot \text{affinity} + w_{\text{pop}} \cdot \frac{\log(1 + \text{clicks}_j)}{\log(1 + \text{max\_clicks})}$$
