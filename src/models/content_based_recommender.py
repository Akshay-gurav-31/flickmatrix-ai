"""
Content-Based Recommender.

Recommends items based on their content features (genres, tags, release year) and
the user's rating history.

Methodology:
    1. Fits a TF-IDF Vectoriser on the 'content_soup' (concatenated genres + tags) of movies.
    2. Builds a User Profile Vector for a target user by computing a weighted average
       of the TF-IDF vectors of movies they have rated, weighted by their ratings (centered around
       the user's mean rating).
    3. Recommends movies by calculating the cosine similarity between the user profile vector
       and all candidate movie TF-IDF vectors.
    4. Finds similar movies by computing cosine similarity directly between movie TF-IDF vectors.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base_recommender import BaseRecommender
from src.utils.exceptions import MovieNotFoundError, UserNotFoundError
from src.utils.helpers import format_recommendation_result
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-Based Filtering Recommender using TF-IDF and Cosine Similarity.

    Attributes:
        movies_df: Enriched movies DataFrame.
        ratings_df: Training ratings DataFrame.
        vectorizer: Fitted TfidfVectorizer instance.
        tfidf_matrix: Sparse matrix of shape (n_movies, n_features).
        movie_id_to_row: Dict mapping movieId to TF-IDF matrix row index.
        row_to_movie_id: Dict mapping TF-IDF matrix row index to movieId.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the ContentBasedRecommender."""
        super().__init__(name="Content-Based Recommender", config_path=config_path)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.cfg.models.content_based.tfidf_max_features,
            ngram_range=tuple(self.cfg.models.content_based.tfidf_ngram_range),
            stop_words="english",
        )
        self.tfidf_matrix: Any = None
        self.movie_id_to_row: Dict[int, int] = {}
        self.row_to_movie_id: Dict[int, int] = {}
        
        self._model_params = {
            "max_features": self.cfg.models.content_based.tfidf_max_features,
            "ngram_range": self.cfg.models.content_based.tfidf_ngram_range,
            "similarity_metric": self.cfg.models.content_based.similarity_metric,
            "top_n_default": self.cfg.models.content_based.top_n_default,
        }

    def _fit(self, data: Any) -> None:
        """
        Fit the TF-IDF vectoriser and map movies to matrix indices.

        Args:
            data: ProcessedData containing movies and ratings.
        """
        self.movies_df = data.movies.copy()
        self.ratings_df = data.train.copy()

        # Build content text corpus
        # 'content_soup' is engineered in preprocessor.py from genre + tags
        content_corpus = self.movies_df["content_soup"].fillna("").tolist()

        logger.info(f"[{self.name}] Fitting TF-IDF Vectoriser on {len(content_corpus)} movies...")
        self.tfidf_matrix = self.vectorizer.fit_transform(content_corpus)
        logger.info(f"[{self.name}] TF-IDF Matrix shape: {self.tfidf_matrix.shape}")

        # Build mappings between movieId and tfidf matrix row index
        for idx, row in self.movies_df.iterrows():
            movie_id = int(row["movieId"])
            self.movie_id_to_row[movie_id] = idx
            self.row_to_movie_id[idx] = movie_id

    def _build_user_profile(self, user_id: int) -> np.ndarray:
        """
        Build a TF-IDF profile vector for a user based on their ratings.

        Profile = Sum of (movie_tfidf_vector * centered_rating)
        Ratings are centered around the user's mean rating to represent
        likes as positive weights and dislikes as negative weights.

        Args:
            user_id: Target user ID.

        Returns:
            Normalized User Profile Vector of shape (1, n_features).
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_ratings.empty:
            raise UserNotFoundError(user_id=user_id)

        user_mean = user_ratings["rating"].mean()
        
        # Filter valid movies present in TF-IDF matrix
        valid_ratings = user_ratings[user_ratings["movieId"].isin(self.movie_id_to_row)].copy()
        if valid_ratings.empty:
            n_features = self.tfidf_matrix.shape[1]
            return np.zeros((1, n_features))

        row_indices = [self.movie_id_to_row[m] for m in valid_ratings["movieId"]]
        ratings = valid_ratings["rating"].values
        weights = ratings - user_mean
        weights = np.where(weights >= 0, weights + 1.0, weights)

        # Matrix product yields user profile vector
        user_profile_sparse = weights.reshape(1, -1) @ self.tfidf_matrix[row_indices]
        user_profile = np.asarray(user_profile_sparse)

        # L2 normalize user profile
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm
            
        return user_profile

    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Recommend items based on cosine similarity with the user's profile.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: If True, exclude movies the user has already rated.

        Returns:
            List of recommendations.
        """
        try:
            user_profile = self._build_user_profile(user_id)
        except UserNotFoundError:
            # Cold-start fallback: return popular items
            logger.warning(f"[{self.name}] Cold-start user {user_id} detected. Falling back to popularity model recommendations.")
            # Dynamic imports inside models to prevent circular dependencies
            from src.models.popularity_recommender import PopularityRecommender
            pop_model = PopularityRecommender(config_path=self.cfg.paths.config_path if hasattr(self.cfg.paths, 'config_path') else None)
            class MockData:
                def __init__(self, train, movies):
                    self.train = train
                    self.movies = movies
            pop_model.fit(MockData(self.ratings_df, self.movies_df))
            return pop_model.recommend(user_id, n, exclude_seen)

        # Compute cosine similarities between user profile and all movies
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # Identify seen movies
        seen_movie_ids = set()
        if exclude_seen:
            seen_movie_ids = set(
                self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
            )

        # Rank movies
        ranked_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in ranked_indices:
            if len(recommendations) >= n:
                break
                
            movie_id = self.row_to_movie_id[idx]
            if exclude_seen and movie_id in seen_movie_ids:
                continue

            score = float(similarities[idx])
            if score <= 0.0:
                # Similarity is 0 or negative; stop recommending content-based match
                break

            movie_row = self.movies_df[self.movies_df["movieId"] == movie_id].iloc[0]
            
            # Find the top features (genres/tags) causing this recommendation
            feature_names = self.vectorizer.get_feature_names_out()
            movie_vector = self.tfidf_matrix[idx].toarray().flatten()
            profile_vector = user_profile.flatten()
            
            # Element-wise product shows which words contributed the most to similarity
            elementwise_product = movie_vector * profile_vector
            top_word_indices = np.argsort(elementwise_product)[::-1][:3]
            top_words = [feature_names[w_idx] for w_idx in top_word_indices if elementwise_product[w_idx] > 0]
            
            genre_list = movie_row["genre_list"]
            matched_genres = [g for g in genre_list if g.lower() in [w.lower() for w in top_words]]
            
            if matched_genres:
                matched_str = " and ".join(matched_genres[:2])
                explanation = f"Recommended because it matches your interest in {matched_str} movies."
            elif top_words:
                matched_str = ", ".join(top_words[:2])
                explanation = f"Recommended because of matching content themes: {matched_str}."
            else:
                explanation = f"Recommended because it matches your preference in similar genre profiles."

            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            recommendations.append(rec)

        # If we couldn't find enough recommendations, pad with popularity model
        if len(recommendations) < n:
            from src.models.popularity_recommender import PopularityRecommender
            pop_model = PopularityRecommender()
            class MockData:
                def __init__(self, train, movies):
                    self.train = train
                    self.movies = movies
            pop_model.fit(MockData(self.ratings_df, self.movies_df))
            pad_recs = pop_model.recommend(user_id, n - len(recommendations), exclude_seen)
            recommendations.extend(pad_recs)

        return recommendations

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict a rating based on content similarity.

        Scores cosine similarity of user profile & movie, then scales it
        to the [0.5, 5.0] range based on user's rating distribution.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating.
        """
        if movie_id not in self.movie_id_to_row:
            return float(self.ratings_df["rating"].mean())

        try:
            user_profile = self._build_user_profile(user_id)
        except UserNotFoundError:
            # User cold start: return global or movie average rating
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return float(match.iloc[0]["bayesian_avg"])
            return float(self.ratings_df["rating"].mean())

        movie_idx = self.movie_id_to_row[movie_id]
        movie_vector = self.tfidf_matrix[movie_idx]
        
        sim = float(cosine_similarity(user_profile, movie_vector)[0][0])
        
        # Scale similarity [0.0, 1.0] to rating scale [min_r, max_r] based on user mean
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        user_mean = float(user_ratings["rating"].mean())
        user_std = float(user_ratings["rating"].std()) if len(user_ratings) > 1 else 1.0
        
        # Map similarity to a rating around user_mean
        predicted = user_mean + (sim - 0.2) * 2.0 * user_std
        from src.utils.helpers import clip_rating
        return clip_rating(predicted, self.cfg.data.rating_scale_min, self.cfg.data.rating_scale_max)

    def _similar_movies(self, movie_id: int, n: int) -> List[Dict[str, Any]]:
        """
        Find similar movies using cosine similarity of TF-IDF vectors.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies to return.

        Returns:
            List of similar movies.
        """
        if movie_id not in self.movie_id_to_row:
            raise MovieNotFoundError(movie_id=movie_id)

        target_row_idx = self.movie_id_to_row[movie_id]
        target_vector = self.tfidf_matrix[target_row_idx]

        # Compute similarity with all movies
        similarities = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        # Sort indices descending
        ranked_indices = np.argsort(similarities)[::-1]
        
        similar_list = []
        for idx in ranked_indices:
            current_movie_id = self.row_to_movie_id[idx]
            if current_movie_id == movie_id:
                continue  # Skip itself

            if len(similar_list) >= n:
                break

            score = float(similarities[idx])
            movie_row = self.movies_df[self.movies_df["movieId"] == current_movie_id].iloc[0]
            
            # Shared genres
            source_movie = self.movies_df[self.movies_df["movieId"] == movie_id].iloc[0]
            shared_genres = set(source_movie["genre_list"]).intersection(set(movie_row["genre_list"]))
            
            if shared_genres:
                genres_str = ", ".join(list(shared_genres)[:2])
                explanation = f"Similar because it shares genres: {genres_str}."
            else:
                explanation = "Similar because it shares tags and content themes."

            rec = self._build_recommendation_dict(
                movie_row=movie_row,
                score=score,
                explanation=explanation,
                poster_url=movie_row.get("poster_url")
            )
            similar_list.append(rec)
            
        return similar_list
