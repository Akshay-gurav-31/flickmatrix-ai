"""
Consolidated Recommendation System Tests.

Contains basic validation tests:
    - Preprocessing year extraction parsing checks.
    - End-to-end Recommender training and recommendation contract integration checks.
"""

import pandas as pd
import pytest

from src.data.preprocessor import ProcessedData
from src.utils.helpers import extract_year_from_title
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based_recommender import ContentBasedRecommender
from src.models.item_cf_recommender import ItemCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.hybrid_recommender import HybridRecommender


def test_extract_year_from_title():
    """Verify year is parsed correctly from title strings."""
    assert extract_year_from_title("Toy Story (1995)") == 1995
    assert extract_year_from_title("Inception (2010)") == 2010
    assert extract_year_from_title("No Year Movie") is None


@pytest.fixture
def mock_processed_data() -> ProcessedData:
    """Create minimal Mock ProcessedData for quick training tests."""
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
                "Unseen Movie (2020)"
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

    user_item = train_ratings.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0)

    return ProcessedData(
        train=train_ratings,
        movies=movies,
        user_item_matrix=user_item,
        item_user_matrix=user_item.T,
        all_movie_ids=[1, 2, 3, 4, 5, 6],
        all_user_ids=[1, 2, 3, 4, 5]
    )


@pytest.mark.parametrize(
    "model_class",
    [
        PopularityRecommender,
        ContentBasedRecommender,
        ItemCFRecommender,
        SVDRecommender,
        HybridRecommender,
    ],
)
def test_recommenders_lifecycle(model_class, mock_processed_data):
    """Verify models train and output valid recommendation structures."""
    model = model_class()
    model.fit(mock_processed_data)
    
    assert model.is_fitted
    
    # Generate recommendations
    recs = model.recommend(user_id=1, n=2, exclude_seen=True)
    assert len(recs) <= 2
    
    if recs:
        rec = recs[0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert "explanation" in rec
        
    # Generate predictions
    pred = model.predict(user_id=1, movie_id=4)
    assert isinstance(pred, float)
    assert 0.5 <= pred <= 5.0
