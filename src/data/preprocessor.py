"""
Data Preprocessing Pipeline for the Recommendation System.

This module transforms raw MovieLens CSV files into model-ready DataFrames
and persists them to data/processed/. It performs:

    1. Data loading with type casting
    2. Quality filtering (min ratings per user/movie)
    3. Feature engineering:
       - Year extraction from titles
       - Genre list parsing
       - Tag aggregation per movie
       - TF-IDF feature vector preparation (text corpus)
    4. User-item rating matrix construction
    5. Train/test splitting with temporal awareness
    6. Saving all processed artifacts as Parquet files

All methods are documented and all DataFrames have clearly named columns.

Usage:
    from src.data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    data = preprocessor.run()
    # data["train"], data["test"], data["movies"], data["user_item_matrix"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.downloader import MovieLensDownloader
from src.utils.exceptions import DataNotFoundError, DataPreprocessingError
from src.utils.helpers import (
    clean_title,
    ensure_dir,
    extract_year_from_title,
    genres_to_string,
    load_config,
    parse_genres,
    resolve_path,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """
    Container for all processed datasets produced by DataPreprocessor.

    Attributes:
        ratings_raw: Original filtered ratings DataFrame.
        train: Training split of ratings (80%).
        test: Test split of ratings (20%).
        movies: Enriched movies DataFrame with year, clean_title, genre_list,
                genre_string, tag_string, and content_soup columns.
        links: MovieLens ↔ TMDB/IMDB ID mapping.
        user_item_matrix: Pivot table (users × movies) of ratings.
                          Filled with 0.0 for unobserved pairs.
        item_user_matrix: Transpose of user_item_matrix.
        movie_id_to_idx: Maps movieId → row index in item matrices.
        idx_to_movie_id: Maps row index → movieId.
        user_id_to_idx: Maps userId → row index in user matrices.
        idx_to_user_id: Maps row index → userId.
        all_movie_ids: Sorted list of all movie IDs in the filtered dataset.
        all_user_ids: Sorted list of all user IDs in the filtered dataset.
        stats: Summary statistics dictionary.
    """

    ratings_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    train: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    movies: pd.DataFrame = field(default_factory=pd.DataFrame)
    links: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_item_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    item_user_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    movie_id_to_idx: Dict[int, int] = field(default_factory=dict)
    idx_to_movie_id: Dict[int, int] = field(default_factory=dict)
    user_id_to_idx: Dict[int, int] = field(default_factory=dict)
    idx_to_user_id: Dict[int, int] = field(default_factory=dict)
    all_movie_ids: list = field(default_factory=list)
    all_user_ids: list = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline for MovieLens data.

    Attributes:
        cfg: OmegaConf configuration.
        processed_dir: Output directory for processed Parquet files.
        downloader: MovieLensDownloader instance.

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> data = preprocessor.run()
        >>> data.train.shape
        (80000, 4)
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialise the preprocessor with configuration.

        Args:
            config_path: Path to config.yaml. Uses default if None.
        """
        self.cfg = load_config(config_path)
        self.processed_dir: Path = resolve_path(self.cfg.paths.processed_dir)
        self.downloader = MovieLensDownloader(config_path)

    def run(self, force_reprocess: bool = False) -> ProcessedData:
        """
        Execute the full preprocessing pipeline.

        Idempotent: If processed files already exist and ``force_reprocess``
        is False, loads from disk instead of reprocessing.

        Args:
            force_reprocess: If True, reprocess even if cached files exist.

        Returns:
            ProcessedData dataclass with all processed artifacts.

        Raises:
            DataPreprocessingError: If any preprocessing step fails.
        """
        if not force_reprocess and self._is_processed():
            logger.info("Processed data found on disk — loading from cache.")
            return self._load_from_disk()

        logger.info("Starting data preprocessing pipeline...")

        # Ensure raw data exists
        self.downloader.download()
        paths = self.downloader.get_data_paths()

        try:
            # ── Step 1: Load raw data ──────────────────────────────────────
            ratings_raw, movies_raw, tags_raw, links = self._load_raw_data(paths)

            # ── Step 2: Filter low-activity users and sparse movies ────────
            ratings_filtered = self._apply_quality_filters(ratings_raw)

            # ── Step 3: Enrich movies with features ────────────────────────
            movies_enriched = self._engineer_movie_features(
                movies_raw, tags_raw, ratings_filtered
            )

            # Filter movies to only those in the filtered ratings
            active_movie_ids = set(ratings_filtered["movieId"].unique())
            movies_enriched = movies_enriched[
                movies_enriched["movieId"].isin(active_movie_ids)
            ].reset_index(drop=True)

            # ── Step 4: Train/Test split ───────────────────────────────────
            train, test = self._temporal_train_test_split(ratings_filtered)

            # ── Step 5: Build user-item matrix ─────────────────────────────
            user_item_matrix = self._build_user_item_matrix(ratings_filtered)

            # ── Step 6: Build index maps ───────────────────────────────────
            movie_id_to_idx, idx_to_movie_id = self._build_index_map(
                sorted(user_item_matrix.columns.tolist())
            )
            user_id_to_idx, idx_to_user_id = self._build_index_map(
                sorted(user_item_matrix.index.tolist())
            )

            # ── Step 7: Compile statistics ─────────────────────────────────
            stats = self._compute_stats(ratings_filtered, train, test, movies_enriched)
            self._log_stats(stats)

            # ── Step 8: Persist to disk ────────────────────────────────────
            data = ProcessedData(
                ratings_raw=ratings_filtered,
                train=train,
                test=test,
                movies=movies_enriched,
                links=links,
                user_item_matrix=user_item_matrix,
                item_user_matrix=user_item_matrix.T,
                movie_id_to_idx=movie_id_to_idx,
                idx_to_movie_id=idx_to_movie_id,
                user_id_to_idx=user_id_to_idx,
                idx_to_user_id=idx_to_user_id,
                all_movie_ids=sorted(active_movie_ids),
                all_user_ids=sorted(user_item_matrix.index.tolist()),
                stats=stats,
            )
            self._save_to_disk(data)

            logger.info("Preprocessing complete.")
            return data

        except Exception as exc:
            raise DataPreprocessingError(
                "Preprocessing pipeline failed.",
                details=str(exc),
            ) from exc

    # =========================================================================
    # Private: Loading
    # =========================================================================

    def _load_raw_data(
        self, paths: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all raw CSV files into typed DataFrames.

        Args:
            paths: Dictionary of file paths from downloader.get_data_paths().

        Returns:
            Tuple of (ratings, movies, tags, links) DataFrames.
        """
        logger.info("Loading raw CSV files...")

        ratings = pd.read_csv(
            paths["ratings"],
            dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        )
        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

        movies = pd.read_csv(paths["movies"], dtype={"movieId": np.int32})

        tags = pd.read_csv(
            paths["tags"],
            dtype={"userId": np.int32, "movieId": np.int32},
        )
        tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")
        # Drop null tags
        tags = tags.dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

        links = pd.read_csv(
            paths["links"],
            dtype={"movieId": np.int32},
        )
        # tmdbId may have NaN values; keep as nullable Int64
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
        links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce")

        logger.info(
            f"Loaded — ratings: {len(ratings):,}, movies: {len(movies):,}, "
            f"tags: {len(tags):,}, links: {len(links):,}"
        )
        return ratings, movies, tags, links

    # =========================================================================
    # Private: Filtering
    # =========================================================================

    def _apply_quality_filters(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Remove users and movies with insufficient interaction history.

        This prevents cold-start noise from polluting the training set and
        improves matrix factorisation convergence.

        Args:
            ratings: Raw ratings DataFrame.

        Returns:
            Filtered ratings DataFrame.
        """
        min_user = self.cfg.data.min_user_ratings
        min_movie = self.cfg.data.min_movie_ratings

        initial_count = len(ratings)

        # Iterative filtering: removing users may make some movies fall below
        # threshold, and vice versa — iterate until stable
        prev_len = -1
        iteration = 0
        filtered = ratings.copy()

        while prev_len != len(filtered):
            prev_len = len(filtered)
            iteration += 1

            # Filter users
            user_counts = filtered["userId"].value_counts()
            active_users = user_counts[user_counts >= min_user].index
            filtered = filtered[filtered["userId"].isin(active_users)]

            # Filter movies
            movie_counts = filtered["movieId"].value_counts()
            active_movies = movie_counts[movie_counts >= min_movie].index
            filtered = filtered[filtered["movieId"].isin(active_movies)]

        removed = initial_count - len(filtered)
        logger.info(
            f"Quality filter ({iteration} iterations): "
            f"{initial_count:,} → {len(filtered):,} ratings "
            f"(removed {removed:,} | {removed/initial_count*100:.1f}%)"
        )
        logger.info(
            f"Active users: {filtered['userId'].nunique():,}, "
            f"Active movies: {filtered['movieId'].nunique():,}"
        )
        return filtered.reset_index(drop=True)

    # =========================================================================
    # Private: Feature Engineering
    # =========================================================================

    def _engineer_movie_features(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enrich the movies DataFrame with computed features.

        Added columns:
            - ``clean_title``: Title without the year suffix.
            - ``year``: Release year as integer (NaN if unavailable).
            - ``genre_list``: Python list of genre strings.
            - ``genre_string``: Space-joined genres for TF-IDF.
            - ``tag_string``: Space-joined aggregated user tags.
            - ``content_soup``: Combined genre + tag string for TF-IDF.
            - ``avg_rating``: Mean rating for the movie.
            - ``num_ratings``: Count of ratings.
            - ``bayesian_avg``: Bayesian-smoothed average rating.

        Args:
            movies: Raw movies DataFrame.
            tags: Raw tags DataFrame.
            ratings: Filtered ratings DataFrame.

        Returns:
            Enriched movies DataFrame.
        """
        logger.info("Engineering movie features...")
        df = movies.copy()

        # ── Title-based features ───────────────────────────────────────────
        df["clean_title"] = df["title"].apply(clean_title)
        df["year"] = df["title"].apply(extract_year_from_title)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # ── Genre features ─────────────────────────────────────────────────
        df["genre_list"] = df["genres"].apply(parse_genres)
        df["genre_string"] = df["genre_list"].apply(genres_to_string)

        # ── Tag aggregation ────────────────────────────────────────────────
        # Aggregate all user tags per movie into a single string
        tag_agg = (
            tags.groupby("movieId")["tag"]
            .apply(lambda tags_list: " ".join(tags_list.tolist()))
            .reset_index()
            .rename(columns={"tag": "tag_string"})
        )
        df = df.merge(tag_agg, on="movieId", how="left")
        df["tag_string"] = df["tag_string"].fillna("")

        # ── Content soup for TF-IDF ────────────────────────────────────────
        # Repeat genre_string 3× to up-weight genres vs. tags
        df["content_soup"] = (
            df["genre_string"] + " "
            + df["genre_string"] + " "
            + df["genre_string"] + " "
            + df["tag_string"]
        ).str.strip()

        # ── Rating statistics ──────────────────────────────────────────────
        rating_stats = ratings.groupby("movieId")["rating"].agg(
            avg_rating="mean", num_ratings="count"
        )
        df = df.merge(rating_stats, on="movieId", how="left")
        df["avg_rating"] = df["avg_rating"].fillna(0.0)
        df["num_ratings"] = df["num_ratings"].fillna(0).astype(int)

        # ── Bayesian average rating ────────────────────────────────────────
        # Bayesian avg = (v/(v+m)) * R + (m/(v+m)) * C
        # v = num_ratings, m = min_votes, R = movie avg, C = global avg
        C = ratings["rating"].mean()                        # global mean
        m_percentile = self.cfg.models.popularity.min_votes_percentile
        m = np.percentile(df["num_ratings"].values, m_percentile)

        df["bayesian_avg"] = (
            (df["num_ratings"] / (df["num_ratings"] + m)) * df["avg_rating"]
            + (m / (df["num_ratings"] + m)) * C
        )

        logger.info(
            f"Feature engineering complete — "
            f"movies with tags: {(df['tag_string'] != '').sum():,}, "
            f"movies with year: {df['year'].notna().sum():,}"
        )
        return df.reset_index(drop=True)

    # =========================================================================
    # Private: Train/Test Split
    # =========================================================================

    def _temporal_train_test_split(
        self, ratings: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings into train and test sets using a temporal strategy.

        Strategy: For each user, their most recent 20% of ratings go into
        the test set. This mimics production conditions where the model
        is trained on historical data and evaluated on future ratings.

        Args:
            ratings: Filtered ratings DataFrame with a 'timestamp' column.

        Returns:
            Tuple of (train_df, test_df).
        """
        logger.info("Splitting data into train/test sets (temporal per-user)...")

        test_size = self.cfg.data.test_size

        train_frames = []
        test_frames = []

        for user_id, user_ratings in ratings.groupby("userId"):
            user_sorted = user_ratings.sort_values("timestamp")
            n_test = max(1, int(len(user_sorted) * test_size))
            train_frames.append(user_sorted.iloc[:-n_test])
            test_frames.append(user_sorted.iloc[-n_test:])

        train = pd.concat(train_frames, ignore_index=True)
        test = pd.concat(test_frames, ignore_index=True)

        logger.info(
            f"Train: {len(train):,} ratings | Test: {len(test):,} ratings "
            f"({len(test)/(len(train)+len(test))*100:.1f}% test)"
        )
        return train, test

    # =========================================================================
    # Private: Matrix Construction
    # =========================================================================

    def _build_user_item_matrix(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the dense user × item rating matrix.

        Missing entries are filled with 0.0 (unrated, not zero-rated).
        The matrix has users as rows and movies as columns.

        Args:
            ratings: Filtered ratings DataFrame.

        Returns:
            DataFrame with shape (n_users, n_movies), filled with 0.0 for
            unobserved pairs.
        """
        logger.info("Building user-item rating matrix...")
        matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0.0,
        )
        logger.info(
            f"User-item matrix shape: {matrix.shape} "
            f"(sparsity: {1 - ratings.shape[0] / (matrix.shape[0] * matrix.shape[1]):.3%})"
        )
        return matrix

    def _build_index_map(self, ids: list) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build bidirectional integer ID ↔ matrix index mappings.

        Args:
            ids: Sorted list of IDs (user or movie).

        Returns:
            Tuple of (id_to_idx, idx_to_id) dictionaries.
        """
        id_to_idx = {id_val: idx for idx, id_val in enumerate(ids)}
        idx_to_id = {idx: id_val for idx, id_val in enumerate(ids)}
        return id_to_idx, idx_to_id

    # =========================================================================
    # Private: Statistics
    # =========================================================================

    def _compute_stats(
        self,
        ratings: pd.DataFrame,
        train: pd.DataFrame,
        test: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> Dict:
        """
        Compute summary statistics for logging and model diagnostics.

        Args:
            ratings: Full filtered ratings.
            train: Training split.
            test: Test split.
            movies: Enriched movies DataFrame.

        Returns:
            Dictionary of descriptive statistics.
        """
        n_users = ratings["userId"].nunique()
        n_movies = ratings["movieId"].nunique()
        n_ratings = len(ratings)
        sparsity = 1 - n_ratings / (n_users * n_movies)
        density = n_ratings / (n_users * n_movies)

        genre_series = movies["genre_list"].explode()
        top_genres = genre_series.value_counts().head(10).to_dict()

        return {
            "n_users": n_users,
            "n_movies": n_movies,
            "n_ratings_total": n_ratings,
            "n_ratings_train": len(train),
            "n_ratings_test": len(test),
            "sparsity": round(sparsity, 6),
            "density": round(density, 6),
            "avg_rating": round(ratings["rating"].mean(), 4),
            "std_rating": round(ratings["rating"].std(), 4),
            "min_rating": float(ratings["rating"].min()),
            "max_rating": float(ratings["rating"].max()),
            "avg_ratings_per_user": round(n_ratings / n_users, 2),
            "avg_ratings_per_movie": round(n_ratings / n_movies, 2),
            "top_genres": top_genres,
            "movies_with_tags": int((movies["tag_string"] != "").sum()),
            "movies_with_year": int(movies["year"].notna().sum()),
        }

    def _log_stats(self, stats: Dict) -> None:
        """Log the dataset statistics in a readable format."""
        logger.info("--- Dataset Statistics ---")
        logger.info(f"  Users      : {stats['n_users']:>8,}")
        logger.info(f"  Movies     : {stats['n_movies']:>8,}")
        logger.info(f"  Ratings    : {stats['n_ratings_total']:>8,}")
        logger.info(f"  Train      : {stats['n_ratings_train']:>8,}")
        logger.info(f"  Test       : {stats['n_ratings_test']:>8,}")
        logger.info(f"  Sparsity   : {stats['sparsity']:>8.4%}")
        logger.info(f"  Avg Rating : {stats['avg_rating']:>8.4f}")
        logger.info("--------------------------")

    # =========================================================================
    # Private: Persistence
    # =========================================================================

    def _save_to_disk(self, data: ProcessedData) -> None:
        """
        Save all processed DataFrames and lookup tables to Parquet files.

        Args:
            data: ProcessedData container with all processed artifacts.
        """
        ensure_dir(self.processed_dir)
        logger.info(f"Saving processed data to: {self.processed_dir}")

        # DataFrames saved as Parquet for fast I/O and schema preservation
        data.ratings_raw.to_parquet(self.processed_dir / "ratings.parquet", index=False)
        data.train.to_parquet(self.processed_dir / "train.parquet", index=False)
        data.test.to_parquet(self.processed_dir / "test.parquet", index=False)
        data.movies.to_parquet(self.processed_dir / "movies.parquet", index=False)
        data.links.to_parquet(self.processed_dir / "links.parquet", index=False)

        # User-item matrix saved as Parquet (preserves column names = movieIds)
        data.user_item_matrix.to_parquet(
            self.processed_dir / "user_item_matrix.parquet"
        )

        logger.info("All processed files saved successfully.")

    def _is_processed(self) -> bool:
        """
        Check whether all processed Parquet files already exist on disk.

        Returns:
            True if all expected Parquet files are present.
        """
        expected = [
            "ratings.parquet",
            "train.parquet",
            "test.parquet",
            "movies.parquet",
            "links.parquet",
            "user_item_matrix.parquet",
        ]
        return all((self.processed_dir / f).exists() for f in expected)

    def _load_from_disk(self) -> ProcessedData:
        """
        Load all processed DataFrames from Parquet files.

        Returns:
            ProcessedData container populated from disk.
        """
        d = self.processed_dir

        ratings_raw = pd.read_parquet(d / "ratings.parquet")
        train = pd.read_parquet(d / "train.parquet")
        test = pd.read_parquet(d / "test.parquet")
        movies = pd.read_parquet(d / "movies.parquet")
        links = pd.read_parquet(d / "links.parquet")
        user_item_matrix = pd.read_parquet(d / "user_item_matrix.parquet")

        # Rebuild index maps from loaded matrix
        movie_ids = sorted(user_item_matrix.columns.tolist())
        user_ids = sorted(user_item_matrix.index.tolist())
        movie_id_to_idx, idx_to_movie_id = self._build_index_map(movie_ids)
        user_id_to_idx, idx_to_user_id = self._build_index_map(user_ids)

        stats = self._compute_stats(ratings_raw, train, test, movies)
        self._log_stats(stats)

        logger.info("Processed data loaded from disk.")
        return ProcessedData(
            ratings_raw=ratings_raw,
            train=train,
            test=test,
            movies=movies,
            links=links,
            user_item_matrix=user_item_matrix,
            item_user_matrix=user_item_matrix.T,
            movie_id_to_idx=movie_id_to_idx,
            idx_to_movie_id=idx_to_movie_id,
            user_id_to_idx=user_id_to_idx,
            idx_to_user_id=idx_to_user_id,
            all_movie_ids=movie_ids,
            all_user_ids=user_ids,
            stats=stats,
        )


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.run()
    print(f"\nTrain shape : {data.train.shape}")
    print(f"Test shape  : {data.test.shape}")
    print(f"Movies shape: {data.movies.shape}")
    print(f"\nMovies columns: {data.movies.columns.tolist()}")
    print(f"\nSample movie:\n{data.movies.iloc[0]}")
