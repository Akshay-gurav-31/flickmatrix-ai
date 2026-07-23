"""
FlickMatrix AI — HuggingFace Spaces Deployment.

Standalone Streamlit app that loads trained ML models directly
(no FastAPI backend required). Designed for HuggingFace Spaces free tier.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.utils.helpers import load_config
from src.data.preprocessor import DataPreprocessor

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FlickMatrix AI",
    page_icon="film",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A1C24;
        border-radius: 4px;
        color: #AEB2C6;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EB1E44 !important;
        color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #EB1E44;
    }
    .movie-card {
        background-color: #1E212A;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #EB1E44;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-size: 18px;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 5px;
    }
    .movie-meta {
        font-size: 13px;
        color: #AEB2C6;
        margin-bottom: 10px;
    }
    .movie-explanation {
        font-size: 14px;
        background-color: #13151D;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #2B2E3A;
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load models and data (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """Load all pre-trained recommendation models from artifacts/models/."""
    models = {}
    model_dir = os.path.join(ROOT_DIR, "artifacts", "models")
    model_names = ["popularity", "content_based", "item_cf", "svd", "hybrid"]
    for name in model_names:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource(show_spinner="Loading movie dataset...")
def load_dataset():
    """Load the preprocessed dataset for search and metadata."""
    cfg = load_config()
    preprocessor = DataPreprocessor(cfg)
    data = preprocessor.run()
    return data


models = load_models()
data = load_dataset()
movies_df = data.movies


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def search_movies(query: str, genre: str = "All", limit: int = 10):
    """Search movies by title substring, optionally filter by genre."""
    mask = movies_df["clean_title"].str.contains(query, case=False, na=False)
    results = movies_df[mask].copy()

    if genre != "All":
        results = results[results["genre_list"].apply(lambda g: genre in g)]

    results = results.head(limit)

    out = []
    for _, row in results.iterrows():
        out.append({
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "clean_title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": row["genre_list"] if isinstance(row["genre_list"], list) else [],
            "bayesian_avg": round(float(row.get("bayesian_avg", 0)), 2),
        })
    return out


def get_similar_movies(movie_id: int, model_name: str, n: int = 10):
    """Get similar movies using the specified model."""
    model = models.get(model_name)
    if model is None:
        return []
    try:
        sims = model.similar_movies(movie_id=movie_id, n=n)
        return sims
    except Exception:
        return []


def get_all_genres():
    """Extract all unique genres from dataset."""
    all_genres = set()
    for gl in movies_df["genre_list"]:
        if isinstance(gl, list):
            all_genres.update(gl)
    return sorted(all_genres)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
logo_path = os.path.join(ROOT_DIR, "assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.markdown("<h2 style='color:#EB1E44; text-align:center;'>FlickMatrix AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.success(f"Models Loaded: {len(models)}/5")

st.sidebar.subheader("System Settings")
selected_model = st.sidebar.selectbox(
    "Active Model",
    options=list(models.keys()),
    format_func=lambda x: x.upper().replace("_", " "),
)

num_recs = st.sidebar.slider(
    "Recommendations Count (N)",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **FlickMatrix AI** is a production-ready Movie Recommendation System.

    *Features:*
    - 4 ML Recommenders
    - Hybrid Blending
    - Personalized Recommendations
    - Explainable AI (XAI)
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <a href="https://github.com/Akshay-gurav-31/flickmatrix-ai" target="_blank" style="text-decoration: none;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px; background-color: #24292e; color: white; padding: 10px 16px; border-radius: 6px; font-weight: 600; margin-top: 10px; border: 1px solid #444c56;">
            <svg height="20" width="20" viewBox="0 0 16 16" fill="white">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            <span>GitHub Repository</span>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>FlickMatrix AI Recommendations</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #AEB2C6; font-size:18px;'>Deploying Hybrid Collaborative & Content Filtering Recommenders</p>", unsafe_allow_html=True)

tab_discover, tab_metrics = st.tabs(
    ["Discover Similar Movies", "Performance Analytics"]
)

# =============================================================================
# Tab 1: Movie Discovery
# =============================================================================
with tab_discover:
    st.subheader("Explore Similar Movies")

    search_query = st.text_input("Search movie title...", value="Inception")

    genres_list = ["All"] + get_all_genres()
    selected_genre = st.selectbox("Filter search results by genre", options=genres_list)

    if search_query:
        results = search_movies(search_query, genre=selected_genre)

        if not results:
            st.info("No matching movies found.")
        else:
            st.markdown("### Search Results (Click Find Similar to explore)")

            for row in results[:5]:
                col_info, col_btn = st.columns([4, 1.5])

                with col_info:
                    year_s = f" ({row['year']})" if row.get("year") else ""
                    st.markdown(f"**{row['clean_title']}{year_s}**")
                    st.markdown(f"Genres: *{', '.join(row['genres'])}* | FlickMatrix Avg: {row['bayesian_avg']}")

                with col_btn:
                    if st.button("Find Similar", key=f"sim_btn_{row['movie_id']}"):
                        with st.spinner("Finding similar matches..."):
                            sims = get_similar_movies(
                                movie_id=row["movie_id"],
                                model_name=selected_model,
                                n=num_recs,
                            )
                            st.session_state["source_movie"] = row["title"]
                            st.session_state["similar_list"] = sims

    # Render similar movies
    if "similar_list" in st.session_state:
        st.write("---")
        st.subheader(f"Movies similar to: '{st.session_state['source_movie']}'")

        sims_recs = st.session_state["similar_list"]

        if not sims_recs:
            st.info("No similar movies found for this model.")
        else:
            for idx, rec in enumerate(sims_recs):
                clean_title = rec["title"]
                year_str = f" ({rec['year']})" if rec.get("year") else ""
                genres_str = ", ".join(rec["genres"])

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">#{idx+1} {clean_title}{year_str}</div>
                        <div class="movie-meta">Similarity: <b>{rec['score']:.2f}</b> | Genres: <i>{genres_str}</i></div>
                        <div class="movie-explanation"><b>Reason:</b> {rec['explanation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# Tab 2: Performance Metrics
# =============================================================================
with tab_metrics:
    st.subheader("Model Benchmark Diagnostics")
    st.markdown(
        """
        Below is a comparison of all trained recommendation algorithms across accuracy
        (RMSE, MAE) and ranking quality (Precision@10).
        """
    )

    metric_data = {
        "Model": [
            "Popularity",
            "Content-Based",
            "Item CF (Collaborative)",
            "SVD (Matrix Factorization)",
            "Hybrid Recommender",
        ],
        "RMSE": [0.9184, 0.9496, 0.8547, 0.8665, 0.8499],
        "MAE": [0.7089, 0.7355, 0.6465, 0.6639, 0.6496],
        "Precision@10": [0.0186, 0.0113, 0.0216, 0.0371, 0.0412],
    }

    metrics_df = pd.DataFrame(metric_data)

    col_card1, col_card2, col_card3 = st.columns(3)
    with col_card1:
        st.metric(
            label="Lowest Prediction Error (Hybrid RMSE)",
            value="0.8499",
            delta="-1.9% improvement over SVD",
            delta_color="normal",
        )
    with col_card2:
        st.metric(
            label="Average Absolute Rating Error (MAE)",
            value="0.6496",
            delta="-2.2% improvement over SVD",
            delta_color="normal",
        )
    with col_card3:
        st.metric(
            label="Top Recommendation Quality (Precision@10)",
            value="4.12%",
            delta="+11.1% over SVD",
        )

    st.write("---")
    metric_choice = st.selectbox(
        "Select Metric to compare",
        options=["RMSE", "MAE", "Precision@10"],
    )

    chart_df = metrics_df.set_index("Model")[[metric_choice]]
    st.bar_chart(chart_df)

    st.markdown("### Benchmark Metrics Summary Table")

    styled_df = metrics_df.copy()
    styled_df["Precision@10"] = styled_df["Precision@10"].apply(lambda x: f"{x:.2%}")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.write("---")
    st.markdown("### Why the Hybrid Model Out-performs Single Models")
    st.markdown(
        """
        1. **Solving the Cold Start & Sparsity Bottleneck:** Collaborative filtering models (Item-CF) and
           Matrix Factorization (SVD) perform exceptionally well when user histories are rich. However, for sparse or
           new profiles, they experience cold-start degradation. The **Hybrid model** detects sparse rating sizes
           dynamically and routes predictions through a content-based + popularity blend, guaranteeing robust fallback scores.
        2. **Multi-Signal Ensembling:** While SVD achieves a lower **RMSE** by mapping ratings to latent factor biases,
           it struggles to capture exact genre preferences. By incorporating **Content-Based TF-IDF** signals directly into
           the final score, the Hybrid model pulls highly-relevant genres to the top of the recommendation list, boosting **Precision@10**.
        """
    )
