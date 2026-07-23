"""
Integration Tests for Recommendation Models.

Trains and evaluates all 6 models on mock datasets:
    - ModelNotTrainedError guard checks
    - Popularity Recommender interface and outputs
    - Content-Based Recommender interface and outputs
    - User CF Recommender interface and outputs
    - Item CF Recommender interface and outputs
    - SVD Recommender interface and outputs
    - Hybrid Recommender ensembling and cold-start fallback
"""

import pandas as pd
import pytest

from src.data.preprocessor import ProcessedData
from src.utils.exceptions import ModelNotTrainedError
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based_recommender import ContentBasedRecommender
# pyrefly: ignore [missing-import]
from src.models.user_cf_recommender import UserCFRecommender
from src.models.item_cf_recommender import ItemCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.hybrid_recommender import HybridRecommender


@pytest.fixture
def mock_processed_data() -> ProcessedData:
    """Provide a minimal Mock ProcessedData set for training recommenders."""
    # 5 users, 6 movies
    train_ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
            "movieId": [1, 2, 3, 1, 4, 2, 3, 5, 1, 5, 2],
            "rating": [5.0, 4.0, 4.5, 4.0, 3.0, 4.0, 5.0, 4.5, 3.5, 2.0, 4.0],
            "timestamp": list(range(1000, 1011)),
        }
    )

    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Heat (1995)",
                "Sabrina (1995)",
                "GoldenEye (1995)",
                "Unseen Movie (2020)"  # unrated
            ],
            "clean_title": ["Toy Story", "Jumanji", "Heat", "Sabrina", "GoldenEye", "Unseen Movie"],
            "year": [1995, 1995, 1995, 1995, 1995, 2020],
            "genre_list": [
                ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
                ["Adventure", "Children", "Fantasy"],
                ["Action", "Crime", "Thriller"],
                ["Comedy", "Romance"],
                ["Action", "Adventure", "Thriller"],
                ["Comedy", "Drama"]
            ],
            "genre_string": [
                "Adventure Animation Children Comedy Fantasy",
                "Adventure Children Fantasy",
                "Action Crime Thriller",
                "Comedy Romance",
                "Action Adventure Thriller",
                "Comedy Drama"
            ],
            "tag_string": ["pixar classic", "board game", "heist pacino", "remake romance", "james bond", ""],
            "content_soup": [
                "Adventure Animation Children Comedy Fantasy pixar classic",
                "Adventure Children Fantasy board game",
                "Action Crime Thriller heist pacino",
                "Comedy Romance remake romance",
                "Action Adventure Thriller james bond",
                "Comedy Drama"
            ],
            "avg_rating": [4.16, 4.0, 4.75, 3.0, 3.25, 0.0],
            "num_ratings": [3, 2, 2, 1, 2, 0],
            "bayesian_avg": [4.0, 3.9, 4.1, 3.2, 3.3, 3.5]
        }
    )

    # Simple lookup matrices
    user_item = train_ratings.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0)

    return ProcessedData(
        train=train_ratings,
        movies=movies,
        user_item_matrix=user_item,
        item_user_matrix=user_item.T,
        all_movie_ids=[1, 2, 3, 4, 5, 6],
        all_user_ids=[1, 2, 3, 4, 5]
    )


# =============================================================================
# Interface and fit assertions
# =============================================================================


def test_untrained_model_raises_error():
    """Verify that calling recommendations before training raises ModelNotTrainedError."""
    model = PopularityRecommender()
    with pytest.raises(ModelNotTrainedError):
        model.recommend(user_id=1, n=5)


@pytest.mark.parametrize(
    "model_class",
    [
        PopularityRecommender,
        ContentBasedRecommender,
        UserCFRecommender,
        ItemCFRecommender,
        SVDRecommender,
        HybridRecommender,
    ],
)
def test_recommenders_lifecycle(model_class, mock_processed_data):
    """Verify all models compile, fit, recommend, predict, and generate similar movies."""
    model = model_class()
    
    # Fit
    model.fit(mock_processed_data)
    assert model.is_fitted

    # Recommend
    recs = model.recommend(user_id=1, n=3, exclude_seen=True)
    assert len(recs) <= 3
    
    # Check output structure
    if recs:
        rec = recs[0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert "explanation" in rec
        assert "genres" in rec

    # Predict
    pred = model.predict(user_id=1, movie_id=4)
    assert isinstance(pred, float)
    assert 0.5 <= pred <= 5.0

    # Similar movies
    sims = model.similar_movies(movie_id=1, n=2)
    assert len(sims) <= 2
    if sims:
        assert sims[0]["movie_id"] != 1  # Should not return itself


def test_hybrid_cold_start_routing(mock_processed_data):
    """Verify that hybrid recommender switches weights for cold-start users."""
    model = HybridRecommender()
    model.fit(mock_processed_data)
    
    # Force threshold and weight configurations inside test
    model.cold_threshold = 4
    
    # User 1 has 3 ratings (< threshold 4) -> Cold start
    weights_cold = model._get_active_weights(user_id=1)
    assert weights_cold["svd"] == 0.0
    assert weights_cold["popularity"] > 0.0

    # User 3 has 3 ratings (< threshold 4) -> Cold start
    # Let's verify that a user with >= threshold ratings gets standard weights
    model.cold_threshold = 2
    weights_standard = model._get_active_weights(user_id=1)  # User 1 has 3 ratings (>= 2)
    assert weights_standard["svd"] > 0.0
