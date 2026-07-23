"""
Hybrid Recommender.

Orchestrates all individual recommendation engines:
    1. Popularity-Based
    2. Content-Based
    3. User Collaborative Filtering
    4. Item Collaborative Filtering
    5. SVD Matrix Factorisation

Combines rating predictions and recommendation lists using a weighted sum approach.
Implements dynamic weighting: switches to cold-start weights for users with very
few interactions, relying on popularity and content rather than sparse CF matrices.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_recommender import BaseRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based_recommender import ContentBasedRecommender
from src.models.item_cf_recommender import ItemCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.utils.exceptions import MovieNotFoundError, UserNotFoundError
from src.utils.helpers import clip_rating, top_n_from_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommender System.

    Attributes:
        movies_df: Enriched movies DataFrame.
        ratings_df: Training ratings DataFrame.
        models: Dictionary containing fitted instances of all sub-models.
        weights: Standard weights used for ensembling.
        cold_weights: Weights used for cold-start users.
        cold_threshold: Minimum rating count required to use standard weights.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the HybridRecommender."""
        super().__init__(name="Hybrid Recommender", config_path=config_path)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        
        # Instantiate sub-models
        self.models: Dict[str, BaseRecommender] = {
            "popularity": PopularityRecommender(config_path),
            "content_based": ContentBasedRecommender(config_path),
            "item_cf": ItemCFRecommender(config_path),
            "svd": SVDRecommender(config_path),
        }
        
        # Load weights from config
        self.weights = dict(self.cfg.models.hybrid.weights)
        self.cold_weights = dict(self.cfg.models.hybrid.cold_start_weights)
        self.cold_threshold = int(self.cfg.models.hybrid.cold_start_threshold)
        
        self._model_params = {
            "weights": self.weights,
            "cold_start_weights": self.cold_weights,
            "cold_start_threshold": self.cold_threshold,
            "top_n_default": self.cfg.models.hybrid.top_n_default,
        }

    def _fit(self, data: Any) -> None:
        """
        Fit all underlying recommender models on the training dataset.

        Args:
            data: ProcessedData containing dataset splits and matrices.
        """
        self.movies_df = data.movies.copy()
        self.ratings_df = data.train.copy()

        # Train each sub-model sequentially
        for key, model in self.models.items():
            model.fit(data)
            
        logger.info(f"[{self.name}] All 5 sub-models successfully fitted.")

    def _get_active_weights(self, user_id: int) -> Dict[str, float]:
        """
        Determine which weights to use based on the user's interaction count.

        If the user is new or has very few ratings (below the cold start threshold),
        we use cold-start weights which rely on content/popularity and zero out SVD/CF.

        Args:
            user_id: Target user ID.

        Returns:
            Dictionary of sub-model name -> float weight.
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        rating_count = len(user_ratings)

        if rating_count < self.cold_threshold:
            logger.debug(
                f"[{self.name}] User {user_id} has {rating_count} ratings (< threshold {self.cold_threshold}). "
                f"Applying COLD-START weights."
            )
            return self.cold_weights
        return self.weights

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating as a weighted combination of all sub-model predictions.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating.
        """
        weights = self._get_active_weights(user_id)
        prediction = 0.0

        for key, model in self.models.items():
            weight = weights.get(key, 0.0)
            if weight == 0.0:
                continue
            
            # Weighted addition of sub-predictions
            pred_score = model.predict(user_id, movie_id)
            prediction += weight * pred_score

        return clip_rating(prediction, self.cfg.data.rating_scale_min, self.cfg.data.rating_scale_max)

    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations by ensembling scores from all models.

        Computes predictions for candidate movies generated from the union
        of recommendations from each sub-model. This keeps execution time low
        compared to scoring the entire catalog.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: If True, exclude already rated items.

        Returns:
            List of hybrid recommendations.
        """
        weights = self._get_active_weights(user_id)
        
        # ── Step 1: Generate candidates pool ───────────────────────────────
        # Pool top-50 candidates from each sub-model with weight > 0
        candidate_ids = set()
        for key, model in self.models.items():
            weight = weights.get(key, 0.0)
            if weight == 0.0:
                continue
            
            # Fetch recommendations from sub-model (requesting n * 3 to broaden candidates)
            try:
                sub_recs = model.recommend(user_id, n=n*3, exclude_seen=exclude_seen)
                for rec in sub_recs:
                    candidate_ids.add(rec["movie_id"])
            except Exception as e:
                logger.warning(f"Sub-model {key} failed to generate candidate recommendations: {e}")

        # If pool is empty, fall back to popularity candidates
        if not candidate_ids:
            pop_recs = self.models["popularity"].recommend(user_id, n=n, exclude_seen=exclude_seen)
            return pop_recs

        # ── Step 2: Score candidates ───────────────────────────────────────
        candidate_scores = {}
        for m_id in candidate_ids:
            candidate_scores[m_id] = self._predict(user_id, m_id)

        # ── Step 3: Sort candidates ────────────────────────────────────────
        top_recs = top_n_from_scores(candidate_scores, n)

        # ── Step 4: Construct results ──────────────────────────────────────
        recommendations = []
        for m_id, score in top_recs:
            movie_row = self.movies_df[self.movies_df["movieId"] == m_id].iloc[0]
            
            # Fetch individual sub-explanations for formatting a hybrid explanation
            sub_reasons = []
            for key in ["content_based", "item_cf"]:
                if weights.get(key, 0.0) > 0.15:
                    try:
                        # Grab specific context from sub-models
                        if key == "content_based":
                            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
                            fav_genres = []
                            if not user_ratings.empty:
                                fav_movies = user_ratings[user_ratings["rating"] >= 4.0]["movieId"].tolist()
                                if fav_movies:
                                    fav_df = self.movies_df[self.movies_df["movieId"].isin(fav_movies)]
                                    fav_genres = fav_df["genre_list"].explode().dropna().value_counts().index.tolist()[:1]
                            overlap = [g for g in movie_row["genre_list"] if g in fav_genres]
                            if overlap:
                                sub_reasons.append(f"it matches your interest in {overlap[0]}")
                        
                        elif key == "item_cf" and m_id in self.models["item_cf"].movie_id_to_idx:
                            # Try to extract the favorite item similarity
                            item_cf_model = self.models["item_cf"]
                            user_favorites = self.ratings_df[(self.ratings_df["userId"] == user_id) & (self.ratings_df["rating"] >= 4.0)]
                            user_favorites = user_favorites.sort_values(by="rating", ascending=False)["movieId"].tolist()
                            if user_favorites:
                                m_idx = item_cf_model.movie_id_to_idx[m_id]
                                fav_indices = [item_cf_model.movie_id_to_idx[fav] for fav in user_favorites if fav in item_cf_model.movie_id_to_idx]
                                if fav_indices:
                                    sim_scores = item_cf_model.item_sim_matrix[m_idx, fav_indices]
                                    best_fav_idx = np.argmax(sim_scores)
                                    if sim_scores[best_fav_idx] > 0.2:
                                        best_fav_id = user_favorites[best_fav_idx]
                                        best_fav_movie = self.movies_df[self.movies_df["movieId"] == best_fav_id].iloc[0]
                                        sub_reasons.append(f"you liked '{best_fav_movie['clean_title']}'")
                    except Exception:
                        pass
                        
            # Format explanation
            if sub_reasons:
                explanation = "Recommended because " + " and ".join(sub_reasons[:2]) + "."
            else:
                explanation = "Recommended based on a balanced blend of popular trends, genre preferences, and collaborative user inputs."

            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            recommendations.append(rec)

        return recommendations

    def _similar_movies(self, movie_id: int, n: int) -> List[Dict[str, Any]]:
        """
        Compute similar movies as a hybrid blend of content, item CF, and SVD.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies.

        Returns:
            List of similar movies.
        """
        # Fetch similarities from content_based, item_cf, and SVD models (if possible)
        candidate_scores = {}
        
        sim_weights = {
            "content_based": 0.40,
            "item_cf": 0.35,
            "svd": 0.25,
        }
        
        for key, weight in sim_weights.items():
            try:
                sub_sims = self.models[key].similar_movies(movie_id, n=n*2)
                for rec in sub_sims:
                    m_id = rec["movie_id"]
                    candidate_scores[m_id] = candidate_scores.get(m_id, 0.0) + (weight * rec["score"])
            except Exception as e:
                logger.warning(f"Could not fetch similarities from model {key}: {e}")

        if not candidate_scores:
            # Fall back to popularity-based similarity
            return self.models["popularity"].similar_movies(movie_id, n)

        # Sort and return top N
        top_recs = top_n_from_scores(candidate_scores, n)

        similar_list = []
        for m_id, score in top_recs:
            movie_row = self.movies_df[self.movies_df["movieId"] == m_id].iloc[0]
            explanation = "Similar because of overlapping genre traits and matching user rating correlations."
            
            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            similar_list.append(rec)

        return similar_list
