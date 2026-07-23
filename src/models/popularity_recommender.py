"""
Popularity-Based Recommender.

Computes recommendations based on overall item popularity.
Uses a Bayesian Average (IMDB weighted rating formula) to prevent items with
very few reviews from dominating the top recommendations.

Formula:
    weighted_score = (v / (v + m)) * R + (m / (v + m)) * C
where:
    v = number of ratings for the movie (num_ratings)
    m = minimum ratings threshold (e.g. 70th percentile of rating counts)
    R = average rating of the movie (avg_rating)
    C = global average rating across all movies

Provides:
    - Base fallback for cold-start users.
    - Standard recommendation and prediction methods (predict returns global or movie average).
    - Similar movies (returns other highly-rated movies in similar genres).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_recommender import BaseRecommender
from src.utils.exceptions import MovieNotFoundError, UserNotFoundError
from src.utils.helpers import format_recommendation_result, top_n_from_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PopularityRecommender(BaseRecommender):
    """
    Popularity-Based Recommender using Bayesian Average weighted scoring.

    Attributes:
        movies_df: Enriched movies DataFrame containing bayesian_avg.
        global_mean: Global average rating (C).
        min_ratings_threshold: Minimum rating counts threshold (m).
        ratings_df: DataFrame of ratings used to identify already rated items.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the PopularityRecommender."""
        super().__init__(name="Popularity Recommender", config_path=config_path)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        self.global_mean: float = 0.0
        self.min_ratings_threshold: float = 0.0
        
        self._model_params = {
            "min_votes_percentile": self.cfg.models.popularity.min_votes_percentile,
            "top_n_default": self.cfg.models.popularity.top_n_default,
        }

    def _fit(self, data: Any) -> None:
        """
        Fit the model by calculating the Bayesian average.

        Args:
            data: ProcessedData object containing movies and ratings.
        """
        # Load from preprocessor data
        self.movies_df = data.movies.copy()
        self.ratings_df = data.train.copy()
        
        # Calculate statistics
        self.global_mean = float(self.ratings_df["rating"].mean())
        
        m_percentile = self._model_params["min_votes_percentile"]
        self.min_ratings_threshold = float(
            np.percentile(self.movies_df["num_ratings"].values, m_percentile)
        )
        
        logger.info(
            f"[{self.name}] Global mean rating (C): {self.global_mean:.4f}. "
            f"Min ratings threshold (m) at {m_percentile}th percentile: {self.min_ratings_threshold:.2f} ratings."
        )

    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Recommend the top-N popular movies.

        Args:
            user_id: Target user ID (unused for training/ranking but checked).
            n: Number of recommendations to return.
            exclude_seen: If True, exclude movies the user has already rated.

        Returns:
            List of recommendation dictionaries.
        """
        # Check if user has rated items if exclude_seen is True
        seen_movie_ids = set()
        if exclude_seen and not self.ratings_df.empty:
            seen_movie_ids = set(
                self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
            )

        # Filter out seen movies and sort by bayesian_avg
        candidates = self.movies_df.copy()
        if seen_movie_ids:
            candidates = candidates[~candidates["movieId"].isin(seen_movie_ids)]
            
        top_candidates = candidates.sort_values(by="bayesian_avg", ascending=False).head(n)
        
        recommendations = []
        for _, row in top_candidates.iterrows():
            explanation = (
                f"Recommended because it is highly rated and popular on FlickMatrix "
                f"({int(row['num_ratings'])} ratings, average rating of {row['avg_rating']:.1f}/5.0)."
            )
            rec = self._build_recommendation_dict(
                movie_row=row,
                score=float(row["bayesian_avg"]),
                explanation=explanation,
                poster_url=row.get("poster_url")
            )
            recommendations.append(rec)
            
        return recommendations

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating. Since this is non-personalized, it returns the movie's
        Bayesian average, falling back to the global average if the movie is unknown.

        Args:
            user_id: User ID (ignored).
            movie_id: Movie ID.

        Returns:
            Predicted rating.
        """
        match = self.movies_df[self.movies_df["movieId"] == movie_id]
        if not match.empty:
            return float(match.iloc[0]["bayesian_avg"])
        return self.global_mean

    def _similar_movies(self, movie_id: int, n: int) -> List[Dict[str, Any]]:
        """
        Find top popular movies in the same genres.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies to return.

        Returns:
            List of similar movies.
        """
        match = self.movies_df[self.movies_df["movieId"] == movie_id]
        if match.empty:
            raise MovieNotFoundError(movie_id=movie_id)
            
        source_movie = match.iloc[0]
        source_genres = set(source_movie["genre_list"])
        
        # Calculate overlap score for other movies based on genre and popularity
        candidates = self.movies_df[self.movies_df["movieId"] != movie_id].copy()
        
        def genre_overlap_score(genres_list: List[str]) -> float:
            if not source_genres or not genres_list:
                return 0.0
            overlap = source_genres.intersection(set(genres_list))
            return len(overlap) / len(source_genres.union(set(genres_list)))
            
        candidates["genre_similarity"] = candidates["genre_list"].apply(genre_overlap_score)
        
        # Rank by genre similarity * bayesian_avg
        # Standardise bayesian_avg between 0 and 1 for weighted multiplication
        min_avg = candidates["bayesian_avg"].min()
        max_avg = candidates["bayesian_avg"].max()
        range_avg = max_avg - min_avg if max_avg != min_avg else 1.0
        
        candidates["norm_bayesian"] = (candidates["bayesian_avg"] - min_avg) / range_avg
        # Weighted score: 70% genre similarity, 30% popular rating
        candidates["similarity_score"] = (candidates["genre_similarity"] * 0.7) + (candidates["norm_bayesian"] * 0.3)
        
        top_similar = candidates.sort_values(by="similarity_score", ascending=False).head(n)
        
        similar_list = []
        for _, row in top_similar.iterrows():
            genres_intersection = source_genres.intersection(set(row["genre_list"]))
            intersection_str = ", ".join(list(genres_intersection)[:3])
            explanation = (
                f"Similar because it shares genres ({intersection_str}) and is highly rated."
            )
            rec = self._build_recommendation_dict(
                movie_row=row,
                score=float(row["similarity_score"]),
                explanation=explanation,
                poster_url=row.get("poster_url")
            )
            similar_list.append(rec)
            
        return similar_list
