"""
SVD Matrix Factorisation Recommender.

Uses the scikit-surprise implementation of Singular Value Decomposition (SVD)
to factorise the user-item rating matrix into user and item latent factors.

Formula:
    pred(u, i) = mu + b_u + b_i + q_i^T * p_u
where:
    mu = global mean rating
    b_u = user bias
    b_i = item bias
    p_u = user latent factor vector
    q_i = item latent factor vector

Provides:
    - Highly accurate ratings predictions.
    - Deep item similarities calculated by comparing item latent vectors (q_i).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base_recommender import BaseRecommender
from src.utils.exceptions import MovieNotFoundError, UserNotFoundError
from src.utils.helpers import clip_rating, top_n_from_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SVDRecommender(BaseRecommender):
    """
    SVD Recommender using scikit-surprise.

    Attributes:
        movies_df: Enriched movies DataFrame.
        ratings_df: Training ratings DataFrame.
        model: Fitted surprise SVD model instance.
        inner_to_raw_items: Mappings from surprise inner IDs to MovieLens movieIds.
        raw_to_inner_items: Mappings from MovieLens movieIds to surprise inner IDs.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the SVDRecommender."""
        super().__init__(name="SVD Matrix Factorization", config_path=config_path)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        
        self.model = SVD(
            n_factors=self.cfg.models.svd.n_factors,
            n_epochs=self.cfg.models.svd.n_epochs,
            lr_all=self.cfg.models.svd.lr_all,
            reg_all=self.cfg.models.svd.reg_all,
            random_state=self.cfg.models.svd.random_state,
        )
        self.inner_to_raw_items: Dict[int, int] = {}
        self.raw_to_inner_items: Dict[int, int] = {}
        
        self._model_params = {
            "n_factors": self.cfg.models.svd.n_factors,
            "n_epochs": self.cfg.models.svd.n_epochs,
            "lr_all": self.cfg.models.svd.lr_all,
            "reg_all": self.cfg.models.svd.reg_all,
            "top_n_default": self.cfg.models.svd.top_n_default,
        }

    def _fit(self, data: Any) -> None:
        """
        Fit the surprise SVD model on the training ratings.

        Args:
            data: ProcessedData containing train split.
        """
        self.movies_df = data.movies.copy()
        self.ratings_df = data.train.copy()

        # Surprise requires a Reader and Dataset wrapper
        reader = Reader(
            rating_scale=(
                self.cfg.data.rating_scale_min,
                self.cfg.data.rating_scale_max
            )
        )
        
        # Load columns: line-by-line order is userId, movieId, rating
        surprise_df = self.ratings_df[["userId", "movieId", "rating"]]
        surprise_data = Dataset.load_from_df(surprise_df, reader)
        
        logger.info(f"[{self.name}] Constructing full surprise trainset...")
        trainset = surprise_data.build_full_trainset()
        
        logger.info(f"[{self.name}] Training SVD model using SGD...")
        self.model.fit(trainset)
        
        # Store surprise inner ID mapping
        self.inner_to_raw_items = {
            inner_id: trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()
        }
        self.raw_to_inner_items = {
            raw_id: inner_id for inner_id, raw_id in self.inner_to_raw_items.items()
        }
        
        logger.info(f"[{self.name}] Model successfully fitted. Number of users: {trainset.n_users}, Items: {trainset.n_items}")

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using the trained SVD parameters.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating.
        """
        # SVD handles unseen users/items natively by falling back to biases + global mean
        pred = self.model.predict(uid=user_id, iid=movie_id)
        return clip_rating(pred.est, self.cfg.data.rating_scale_min, self.cfg.data.rating_scale_max)

    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Recommend items using predicted rating scores.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: If True, filter out already rated items.

        Returns:
            List of recommendations.
        """
        # Find already rated movies by this user
        seen_movie_ids = set()
        if exclude_seen:
            seen_movie_ids = set(
                self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
            )

        candidate_scores = {}
        all_movie_ids = self.movies_df["movieId"].tolist()
        
        # Fast vectorized matrix dot-product for Surprise SVD
        try:
            u_inner = self.model.trainset.to_inner_uid(user_id)
            pu = self.model.pu[u_inner]
            bu = self.model.bu[u_inner]
            global_mean = self.model.trainset.global_mean
            
            for m_id in all_movie_ids:
                if exclude_seen and m_id in seen_movie_ids:
                    continue
                try:
                    i_inner = self.model.trainset.to_inner_iid(m_id)
                    qi = self.model.qi[i_inner]
                    bi = self.model.bi[i_inner]
                    pred = global_mean + bu + bi + np.dot(qi, pu)
                except ValueError:
                    pred = self._predict(user_id, m_id)
                
                candidate_scores[m_id] = clip_rating(
                    pred, self.cfg.data.rating_scale_min, self.cfg.data.rating_scale_max
                )
        except ValueError:
            # Fallback for unknown users
            for m_id in all_movie_ids:
                if exclude_seen and m_id in seen_movie_ids:
                    continue
                candidate_scores[m_id] = self._predict(user_id, m_id)

        # Retrieve top items
        top_recs = top_n_from_scores(candidate_scores, n)

        # Compile explanations using movie characteristics matching user history
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        fav_genres = []
        if not user_ratings.empty:
            # Users favorite genres from high-rated movies
            fav_movies = user_ratings[user_ratings["rating"] >= 4.0]["movieId"].tolist()
            if fav_movies:
                fav_df = self.movies_df[self.movies_df["movieId"].isin(fav_movies)]
                fav_genres = fav_df["genre_list"].explode().dropna().value_counts().index.tolist()[:2]

        recommendations = []
        for m_id, score in top_recs:
            movie_row = self.movies_df[self.movies_df["movieId"] == m_id].iloc[0]
            
            # Form an explanation using user favorite genre match
            genres = movie_row["genre_list"]
            overlap_genres = [g for g in genres if g in fav_genres]
            
            if overlap_genres:
                explanation = f"Recommended because it aligns with your taste in {overlap_genres[0]} movies."
            else:
                explanation = "Recommended because it matches latent attributes aligned with your historical preferences."

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
        Find similar movies by computing cosine similarities between SVD item latent vectors (qi).

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies to return.

        Returns:
            List of similar movies.
        """
        if movie_id not in self.raw_to_inner_items:
            raise MovieNotFoundError(movie_id=movie_id)

        # Retrieve item latent factors (qi) matrix
        item_vectors = self.model.qi  # Shape: (n_items, n_factors)
        
        target_inner_id = self.raw_to_inner_items[movie_id]
        target_vector = item_vectors[target_inner_id].reshape(1, -1)

        # Compute similarity between target latent factor and all other items
        similarities = cosine_similarity(target_vector, item_vectors).flatten()

        ranked_indices = np.argsort(similarities)[::-1]
        
        similar_list = []
        for idx in ranked_indices:
            current_raw_id = self.inner_to_raw_items[idx]
            if current_raw_id == movie_id:
                continue

            if len(similar_list) >= n:
                break

            score = float(similarities[idx])
            
            movie_row = self.movies_df[self.movies_df["movieId"] == current_raw_id].iloc[0]
            explanation = "Similar because it shares complex latent attributes with the source movie."
            
            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            similar_list.append(rec)

        return similar_list
