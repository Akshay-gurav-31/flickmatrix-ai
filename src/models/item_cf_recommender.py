"""
Item-Based Collaborative Filtering Recommender.

Recommends movies by calculating similarities between movies based on user ratings.

Methodology:
    1. Creates a user-item rating matrix.
    2. Computes adjusted cosine similarity between items (subtracting user average ratings
       to remove user bias).
    3. Predicts ratings using a weighted average of the user's ratings for similar items:
       pred(u, i) = sum(sim(i, j) * r(u, j)) / sum(|sim(i, j)|)
       for top-k items j rated by user u.
    4. Supports an Implicit Feedback mode (bonus requirement) where interactions are treated
       as binary values, and scores are computed as the sum of similarities to watched movies.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base_recommender import BaseRecommender
from src.utils.exceptions import MovieNotFoundError, UserNotFoundError
from src.utils.helpers import clip_rating, top_n_from_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ItemCFRecommender(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender.

    Attributes:
        movies_df: Enriched movies DataFrame.
        ratings_df: Training ratings DataFrame.
        user_item_matrix: Pivoted user-item rating matrix.
        item_sim_matrix: Cosine similarity matrix between items of shape (n_items, n_items).
        movie_id_to_idx: Bidirectional index lookups.
        user_means: Mean rating for each user.
        implicit_mode: If True, uses binary implicit feedback scoring.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the ItemCFRecommender."""
        super().__init__(name="Item Collaborative Filtering", config_path=config_path)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        self.user_item_matrix: pd.DataFrame = pd.DataFrame()
        
        self.item_sim_matrix: np.ndarray = np.empty((0, 0))
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self.user_means: pd.Series = pd.Series(dtype=np.float32)
        
        self._model_params = {
            "similarity_metric": self.cfg.models.item_cf.similarity_metric,
            "k_neighbors": self.cfg.models.item_cf.k_neighbors,
            "min_common_users": self.cfg.models.item_cf.min_common_users,
            "top_n_default": self.cfg.models.item_cf.top_n_default,
        }

    def _fit(self, data: Any) -> None:
        """
        Fit the item-based CF model. Computes the adjusted item similarity matrix.

        Args:
            data: ProcessedData containing train split.
        """
        self.movies_df = data.movies.copy()
        self.ratings_df = data.train.copy()

        # Build user-item matrix
        logger.info(f"[{self.name}] Constructing user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0.0
        )
        
        # Setup mappings
        movie_ids = sorted(self.user_item_matrix.columns.tolist())
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
        self.idx_to_movie_id = {idx: mid for idx, mid in enumerate(movie_ids)}

        # Compute user means for adjusted cosine similarity
        self.user_means = self.ratings_df.groupby("userId")["rating"].mean()

        logger.info(f"[{self.name}] Fitting in Explicit (Adjusted Cosine) Mode...")
        # Create user-centered rating matrix
        centered_matrix = self.user_item_matrix.copy()
        for user_id, mean_r in self.user_means.items():
            mask = centered_matrix.loc[user_id] > 0.0
            centered_matrix.loc[user_id, mask] = centered_matrix.loc[user_id, mask] - mean_r
            
        # Cosine similarity of centered items (columns)
        self.item_sim_matrix = cosine_similarity(centered_matrix.values.T)

        # Set self-similarity to 0.0
        np.fill_diagonal(self.item_sim_matrix, 0.0)
        logger.info(f"[{self.name}] Fitted item similarity matrix: {self.item_sim_matrix.shape}")

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using item-to-item similarity weighted average.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating.
        """
        # Fallbacks
        if user_id not in self.user_item_matrix.index:
            # Fallback to movie average
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return float(match.iloc[0]["bayesian_avg"])
            return float(self.ratings_df["rating"].mean())

        if movie_id not in self.movie_id_to_idx:
            # Fallback to user mean
            if user_id in self.user_means:
                return float(self.user_means[user_id])
            return float(self.ratings_df["rating"].mean())

        movie_idx = self.movie_id_to_idx[movie_id]

        # Ratings user has given to all movies
        user_ratings = self.user_item_matrix.loc[user_id].values
        
        # Find index of movies rated by user
        rated_movie_indices = np.where(user_ratings > 0.0)[0]
        if len(rated_movie_indices) == 0:
            return float(self.user_means[user_id])

        # Get similarity of movie_id with all rated movies of this user
        similarities = self.item_sim_matrix[movie_idx, rated_movie_indices]
        
        # Keep positive similarities
        pos_indices = np.where(similarities > 0.0)[0]
        if len(pos_indices) == 0:
            return float(self.user_means[user_id])
            
        similarities = similarities[pos_indices]
        rated_movie_indices = rated_movie_indices[pos_indices]

        # Sort similarities to keep top-K neighbors
        k = self._model_params["k_neighbors"]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        similarities = similarities[top_k_indices]
        rated_movie_indices = rated_movie_indices[top_k_indices]

        # Weighted average rating prediction
        sim_sum = np.sum(np.abs(similarities))
        if sim_sum == 0.0:
            return float(self.user_means[user_id])

        ratings_to_neighbors = user_ratings[rated_movie_indices]
        
        pred = np.sum(similarities * ratings_to_neighbors) / sim_sum
        return clip_rating(pred, self.cfg.data.rating_scale_min, self.cfg.data.rating_scale_max)

    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Recommend items using item CF.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: If True, exclude already rated items.

        Returns:
            List of recommendations.
        """
        if user_id not in self.user_item_matrix.index:
            logger.warning(f"[{self.name}] User {user_id} not found in training data. Falling back to popularity recommender.")
            from src.models.popularity_recommender import PopularityRecommender
            pop = PopularityRecommender()
            class MockData:
                def __init__(self, train, movies):
                    self.train = train
                    self.movies = movies
            pop.fit(MockData(self.ratings_df, self.movies_df))
            return pop.recommend(user_id, n, exclude_seen)

        # Retrieve movies user rated with high ratings (>= 3.0)
        user_ratings_series = self.user_item_matrix.loc[user_id]
        rated_movie_ids = user_ratings_series[user_ratings_series > 0.0].index.tolist()
        seen_movie_ids = set(rated_movie_ids)
        
        # User history sorted to explain later
        user_favorites = self.ratings_df[(self.ratings_df["userId"] == user_id) & (self.ratings_df["rating"] >= 4.0)]
        user_favorites = user_favorites.sort_values(by="rating", ascending=False)["movieId"].tolist()

        # Vectorized item-CF prediction across all catalog movies
        user_ratings = self.user_item_matrix.loc[user_id].values
        user_mean = float(self.user_means.get(user_id, self.ratings_df["rating"].mean()))
        rated_mask = (user_ratings > 0.0).astype(float)
        
        # Center ratings by user mean
        centered_ratings = np.where(user_ratings > 0.0, user_ratings - user_mean, 0.0)
        
        sim_sums = self.item_sim_matrix @ rated_mask
        weighted_sums = self.item_sim_matrix @ centered_ratings
        
        predicted_ratings = np.where(
            sim_sums > 0.0,
            user_mean + (weighted_sums / np.maximum(sim_sums, 1e-9)),
            user_mean
        )
        predicted_ratings = np.clip(
            predicted_ratings,
            self.cfg.data.rating_scale_min,
            self.cfg.data.rating_scale_max
        )

        candidate_scores = {}
        all_movie_ids = self.user_item_matrix.columns.tolist()

        for idx, movie_id in enumerate(all_movie_ids):
            if exclude_seen and movie_id in seen_movie_ids:
                continue
            pred = float(predicted_ratings[idx])
            if pred >= user_mean:
                candidate_scores[movie_id] = pred

        top_recs = top_n_from_scores(candidate_scores, n)

        recommendations = []
        for m_id, score in top_recs:
            movie_row = self.movies_df[self.movies_df["movieId"] == m_id].iloc[0]
            
            # Find the favorite movie of the user that is most similar to this recommended movie
            explanation = "Recommended because it is highly similar to items in your watch history."
            if user_favorites and m_id in self.movie_id_to_idx:
                m_idx = self.movie_id_to_idx[m_id]
                
                # Check user favorites similarities
                fav_indices = [self.movie_id_to_idx[fav] for fav in user_favorites if fav in self.movie_id_to_idx]
                if fav_indices:
                    sim_scores = self.item_sim_matrix[m_idx, fav_indices]
                    best_fav_idx = np.argmax(sim_scores)
                    if sim_scores[best_fav_idx] > 0.0:
                        best_fav_id = user_favorites[best_fav_idx]
                        best_fav_movie = self.movies_df[self.movies_df["movieId"] == best_fav_id].iloc[0]
                        explanation = f"Recommended because you liked '{best_fav_movie['clean_title']}'."

            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            recommendations.append(rec)

        # Pad with popularity if needed
        if len(recommendations) < n:
            from src.models.popularity_recommender import PopularityRecommender
            pop = PopularityRecommender()
            class MockData:
                def __init__(self, train, movies):
                    self.train = train
                    self.movies = movies
            pop.fit(MockData(self.ratings_df, self.movies_df))
            pad_recs = pop.recommend(user_id, n - len(recommendations), exclude_seen)
            recommendations.extend(pad_recs)

        return recommendations

    def _similar_movies(self, movie_id: int, n: int) -> List[Dict[str, Any]]:
        """
        Find similar movies directly from item similarity matrix.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies.

        Returns:
            List of similar movies.
        """
        if movie_id not in self.movie_id_to_idx:
            raise MovieNotFoundError(movie_id=movie_id)

        target_idx = self.movie_id_to_idx[movie_id]
        similarities = self.item_sim_matrix[target_idx]

        ranked_indices = np.argsort(similarities)[::-1]
        
        similar_list = []
        for idx in ranked_indices:
            current_m_id = self.idx_to_movie_id[idx]
            if current_m_id == movie_id:
                continue

            if len(similar_list) >= n:
                break

            score = float(similarities[idx])
            if score <= 0.0:
                break

            movie_row = self.movies_df[self.movies_df["movieId"] == current_m_id].iloc[0]
            explanation = "Similar because users who rated one movie also rated this movie in a matching pattern."
            
            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            similar_list.append(rec)

        return similar_list
