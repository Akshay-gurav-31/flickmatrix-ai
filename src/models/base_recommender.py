"""
Abstract Base Recommender Class.

Defines the interface contract that every recommendation algorithm must
implement. This enforces consistency across all models and enables the
Hybrid Recommender to treat all algorithms interchangeably.

All concrete recommenders must implement:
    - fit(data): Train the model on processed data
    - recommend(user_id, n, exclude_seen): Return top-N recommendations
    - predict(user_id, movie_id): Predict rating for a user-movie pair
    - similar_movies(movie_id, n): Return N most similar movies
    - save(path): Persist model to disk
    - load(path): Restore model from disk

Design Patterns Used:
    - Template Method: ``fit`` calls abstract ``_fit``, adding shared logging
    - Strategy: Each subclass is a different recommendation strategy
    - Factory: ModelFactory (in trainer.py) creates instances by name
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.utils.exceptions import ModelLoadError, ModelNotTrainedError, ModelSaveError
from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.

    Subclasses must implement all abstract methods. Shared functionality
    (logging, timing, persistence scaffolding) lives here to avoid
    code duplication across six model implementations.

    Attributes:
        name: Human-readable model identifier (e.g. "SVD Recommender").
        cfg: OmegaConf configuration object.
        is_fitted: Boolean flag set to True after successful training.
        _fit_time_seconds: Time taken to fit the model (set by fit()).
        _model_params: Dictionary of hyperparameters used for this model.
    """

    def __init__(self, name: str, config_path: Optional[str] = None) -> None:
        """
        Initialise the base recommender.

        Args:
            name: Human-readable model name.
            config_path: Path to config.yaml. Uses default if None.
        """
        self.name = name
        self.cfg = load_config(config_path)
        self.is_fitted: bool = False
        self._fit_time_seconds: float = 0.0
        self._model_params: Dict[str, Any] = {}

    # =========================================================================
    # Public Template Methods
    # =========================================================================

    def fit(self, data: Any) -> "BaseRecommender":
        """
        Train the recommender on the processed dataset.

        This is a template method — it wraps the abstract ``_fit`` with
        shared logging and timing logic so subclasses don't repeat this.

        Args:
            data: ProcessedData object from DataPreprocessor.

        Returns:
            Self (for method chaining).
        """
        logger.info(f"[{self.name}] Starting training...")
        start = time.perf_counter()

        self._fit(data)

        self._fit_time_seconds = time.perf_counter() - start
        self.is_fitted = True
        logger.info(
            f"[{self.name}] Training complete in {self._fit_time_seconds:.2f}s"
        )
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate top-N movie recommendations for a given user.

        Args:
            user_id: Target user ID.
            n: Number of recommendations to return.
            exclude_seen: If True, exclude movies the user has already rated.

        Returns:
            List of recommendation dictionaries with at minimum:
                {"movie_id": int, "title": str, "score": float,
                 "genres": list, "year": int, "explanation": str}

        Raises:
            ModelNotTrainedError: If called before fit().
        """
        self._check_is_fitted()
        return self._recommend(user_id=user_id, n=n, exclude_seen=exclude_seen)

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give to a specific movie.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating as a float in [0.5, 5.0].

        Raises:
            ModelNotTrainedError: If called before fit().
        """
        self._check_is_fitted()
        return self._predict(user_id=user_id, movie_id=movie_id)

    def similar_movies(
        self, movie_id: int, n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find the N most similar movies to the given movie.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies to return.

        Returns:
            List of similar movie dictionaries with at minimum:
                {"movie_id": int, "title": str, "score": float,
                 "genres": list, "year": int, "explanation": str}

        Raises:
            ModelNotTrainedError: If called before fit().
        """
        self._check_is_fitted()
        return self._similar_movies(movie_id=movie_id, n=n)

    def save(self, path: Optional[str] = None) -> Path:
        """
        Serialise the trained model to disk using joblib.

        Args:
            path: File path for the saved model. Defaults to
                  ``artifacts/models/{model_slug}.joblib``.

        Returns:
            Path where the model was saved.

        Raises:
            ModelNotTrainedError: If called before fit().
            ModelSaveError: If serialisation fails.
        """
        self._check_is_fitted()

        if path is None:
            from src.utils.helpers import get_model_path
            save_path = get_model_path(self._get_model_slug())
        else:
            save_path = Path(path)

        ensure_dir(save_path.parent)

        try:
            joblib.dump(self, save_path, compress=3)
            logger.info(f"[{self.name}] Model saved to: {save_path}")
            return save_path
        except Exception as exc:
            raise ModelSaveError(
                f"Failed to save {self.name}",
                details=str(exc),
            ) from exc

    @classmethod
    def load(cls, path: str) -> "BaseRecommender":
        """
        Load a serialised model from disk.

        Args:
            path: Path to the joblib file.

        Returns:
            Loaded recommender instance (with is_fitted=True).

        Raises:
            ModelLoadError: If the file doesn't exist or can't be loaded.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise ModelLoadError(
                f"Model file not found: {load_path}",
                details=f"Run training first to generate {load_path.name}",
            )

        try:
            model = joblib.load(load_path)
            logger.info(f"[{model.name}] Loaded from: {load_path}")
            return model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from: {load_path}",
                details=str(exc),
            ) from exc

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _fit(self, data: Any) -> None:
        """
        Internal training logic. Called by the public ``fit`` template method.

        Args:
            data: ProcessedData object with train/test splits, matrices, movies.
        """
        ...

    @abstractmethod
    def _recommend(
        self, user_id: int, n: int, exclude_seen: bool
    ) -> List[Dict[str, Any]]:
        """
        Internal recommendation logic. Called by the public ``recommend`` method.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: Whether to exclude already-rated movies.

        Returns:
            List of recommendation dictionaries.
        """
        ...

    @abstractmethod
    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Internal rating prediction. Called by the public ``predict`` method.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating as float.
        """
        ...

    @abstractmethod
    def _similar_movies(self, movie_id: int, n: int) -> List[Dict[str, Any]]:
        """
        Internal similar-movies logic. Called by the public ``similar_movies`` method.

        Args:
            movie_id: Source movie ID.
            n: Number of similar movies.

        Returns:
            List of similar movie dictionaries.
        """
        ...

    # =========================================================================
    # Shared Utility Methods (available to all subclasses)
    # =========================================================================

    def _check_is_fitted(self) -> None:
        """
        Raise ModelNotTrainedError if the model has not been trained.

        This guard is called at the start of every public inference method.
        """
        if not self.is_fitted:
            raise ModelNotTrainedError(
                f"{self.name} has not been trained yet.",
                details="Call .fit(data) before calling inference methods.",
            )

    def _get_model_slug(self) -> str:
        """
        Return a filesystem-safe slug for this model (used in artifact naming).

        Returns:
            Lowercase underscore-separated model name.
        """
        return self.name.lower().replace(" ", "_")

    def get_params(self) -> Dict[str, Any]:
        """
        Return the model's hyperparameters for MLflow logging.

        Returns:
            Dictionary of parameter name → value.
        """
        return self._model_params.copy()

    def get_training_info(self) -> Dict[str, Any]:
        """
        Return a dictionary with model training metadata.

        Returns:
            Dictionary with model name, fit status, and training time.
        """
        return {
            "model_name": self.name,
            "is_fitted": self.is_fitted,
            "fit_time_seconds": round(self._fit_time_seconds, 3),
            "params": self.get_params(),
        }

    def _build_recommendation_dict(
        self,
        movie_row: pd.Series,
        score: float,
        explanation: str,
        poster_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardised recommendation dictionary from a movie row.

        This is a convenience method used by all subclasses to ensure
        consistent output format regardless of algorithm.

        Args:
            movie_row: A row from the enriched movies DataFrame.
            score: Recommendation score.
            explanation: Human-readable explanation string.
            poster_url: Optional TMDB poster URL.

        Returns:
            Standardised recommendation dictionary.
        """
        from src.utils.helpers import format_recommendation_result

        return format_recommendation_result(
            movie_id=int(movie_row["movieId"]),
            title=str(movie_row.get("clean_title", movie_row.get("title", ""))),
            score=float(score),
            genres=list(movie_row.get("genre_list", [])),
            year=(
                int(movie_row["year"])
                if pd.notna(movie_row.get("year"))
                else None
            ),
            explanation=explanation,
            poster_url=poster_url,
        )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"
