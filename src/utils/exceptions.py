"""
Custom exception hierarchy for the Recommendation System.

All application-specific errors inherit from ``RecommendationSystemError``
so callers can catch the base class or handle specific sub-types.

Exception Tree:
    RecommendationSystemError
    ├── DataError
    │   ├── DataDownloadError
    │   ├── DataNotFoundError
    │   └── DataPreprocessingError
    ├── ModelError
    │   ├── ModelNotTrainedError
    │   ├── ModelLoadError
    │   ├── ModelSaveError
    │   └── ModelPredictionError
    ├── EvaluationError
    ├── ConfigError
    └── APIError
        ├── UserNotFoundError
        └── MovieNotFoundError
"""

from typing import Optional


# =============================================================================
# Base Exception
# =============================================================================


class RecommendationSystemError(Exception):
    """
    Base exception for all recommendation system errors.

    All custom exceptions inherit from this class, allowing broad
    or granular exception handling throughout the application.
    """

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        """
        Initialise the exception with a message and optional details.

        Args:
            message: Human-readable error description.
            details: Additional context (stack trace, file path, etc.).
        """
        self.message = message
        self.details = details
        full_message = f"{message}" + (f" | Details: {details}" if details else "")
        super().__init__(full_message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# =============================================================================
# Data Exceptions
# =============================================================================


class DataError(RecommendationSystemError):
    """Base class for all data-related errors."""


class DataDownloadError(DataError):
    """
    Raised when the dataset cannot be downloaded from the remote URL.

    Example:
        raise DataDownloadError(
            "Failed to download MovieLens dataset",
            details="HTTP 503 from https://files.grouplens.org/..."
        )
    """


class DataNotFoundError(DataError):
    """
    Raised when expected data files are missing from disk.

    Example:
        raise DataNotFoundError(
            "ratings.csv not found",
            details="Expected at data/raw/ml-latest-small/ratings.csv"
        )
    """


class DataPreprocessingError(DataError):
    """
    Raised when data cleaning or feature engineering fails.

    Example:
        raise DataPreprocessingError(
            "Failed to merge ratings with movies",
            details="Column 'movieId' missing from ratings DataFrame"
        )
    """


# =============================================================================
# Model Exceptions
# =============================================================================


class ModelError(RecommendationSystemError):
    """Base class for all model-related errors."""


class ModelNotTrainedError(ModelError):
    """
    Raised when a recommender's predict/recommend method is called
    before the model has been trained.

    Example:
        raise ModelNotTrainedError(
            "SVDRecommender has not been trained yet",
            details="Call .fit() before calling .recommend()"
        )
    """


class ModelLoadError(ModelError):
    """
    Raised when a serialised model artifact cannot be loaded from disk.

    Example:
        raise ModelLoadError(
            "Failed to load SVD model",
            details="File artifacts/models/svd_recommender.joblib not found"
        )
    """


class ModelSaveError(ModelError):
    """
    Raised when a model cannot be serialised and saved to disk.

    Example:
        raise ModelSaveError(
            "Failed to save hybrid model",
            details="Permission denied: artifacts/models/"
        )
    """


class ModelPredictionError(ModelError):
    """
    Raised when model inference fails at runtime.

    Example:
        raise ModelPredictionError(
            "SVD prediction failed for user_id=999",
            details="User not in training set and no fallback configured"
        )
    """


# =============================================================================
# Evaluation Exceptions
# =============================================================================


class EvaluationError(RecommendationSystemError):
    """
    Raised when metric computation fails.

    Example:
        raise EvaluationError(
            "Cannot compute NDCG@K with K=0",
            details="K must be a positive integer"
        )
    """


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigError(RecommendationSystemError):
    """
    Raised when the configuration file is missing, malformed, or contains
    invalid values.

    Example:
        raise ConfigError(
            "Invalid hybrid weights in config.yaml",
            details="Weights must sum to 1.0, got 0.85"
        )
    """


# =============================================================================
# API Exceptions
# =============================================================================


class APIError(RecommendationSystemError):
    """Base class for API-level errors returned to clients."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, details)


class UserNotFoundError(APIError):
    """
    Raised when the requested user_id does not exist in the system.

    Example:
        raise UserNotFoundError(
            "User 9999 not found in training data",
            status_code=404
        )
    """

    def __init__(self, user_id: int, details: Optional[str] = None) -> None:
        super().__init__(
            message=f"User with id={user_id} not found in the system.",
            status_code=404,
            details=details,
        )
        self.user_id = user_id


class MovieNotFoundError(APIError):
    """
    Raised when the requested movie title or movie_id does not exist.

    Example:
        raise MovieNotFoundError(
            movie_id=99999,
            details="No movie with this ID in MovieLens dataset"
        )
    """

    def __init__(
        self,
        movie_id: Optional[int] = None,
        title: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        identifier = f"id={movie_id}" if movie_id else f"title='{title}'"
        super().__init__(
            message=f"Movie ({identifier}) not found in the system.",
            status_code=404,
            details=details,
        )
        self.movie_id = movie_id
        self.title = title
