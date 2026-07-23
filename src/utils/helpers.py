"""
Shared utility functions for the Recommendation System.

Responsibilities:
    - Load and validate the OmegaConf configuration
    - Fetch movie posters from the TMDB API with LRU caching
    - Genre parsing helpers
    - Year extraction from MovieLens title strings
    - Model artifact path resolution

All functions are pure (no side effects) unless explicitly documented.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Project root (3 levels up from src/utils/helpers.py) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# =============================================================================
# Configuration
# =============================================================================


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load and return the OmegaConf DictConfig from the YAML configuration file.

    The result is cached after the first call so subsequent calls are O(1).
    Use ``load_config.cache_clear()`` in tests to reset.

    Args:
        config_path: Path to the YAML config file. Defaults to
                     ``configs/config.yaml`` relative to the project root.

    Returns:
        An OmegaConf DictConfig object with dot-access support.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed.
    """
    from src.utils.exceptions import ConfigError

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}",
            details=f"Expected at {path.absolute()}",
        )

    try:
        cfg = OmegaConf.load(str(path))
        logger.info(f"Configuration loaded from: {path}")
        return cfg
    except Exception as exc:
        raise ConfigError(
            f"Failed to parse configuration file: {path}",
            details=str(exc),
        ) from exc


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root to an absolute Path.

    Args:
        relative_path: Path string relative to project root.

    Returns:
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path


# =============================================================================
# MovieLens Title Parsing
# =============================================================================

# MovieLens titles are formatted as "Movie Title (YEAR)" e.g. "Toy Story (1995)"
_YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
_TITLE_CLEAN_PATTERN = re.compile(r"\s*\(\d{4}\)\s*$")


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract the release year from a MovieLens-formatted movie title.

    Args:
        title: Movie title string, e.g. "Inception (2010)".

    Returns:
        Integer year (e.g. 2010) or None if not found.
    """
    match = _YEAR_PATTERN.search(title)
    if match:
        return int(match.group(1))
    return None


def clean_title(title: str) -> str:
    """
    Remove the year suffix from a MovieLens-formatted title.

    Args:
        title: Movie title string, e.g. "Inception (2010)".

    Returns:
        Clean title string, e.g. "Inception".
    """
    return _TITLE_CLEAN_PATTERN.sub("", title).strip()


def parse_genres(genres_str: str) -> List[str]:
    """
    Parse the pipe-delimited genre string from MovieLens movies.csv.

    Args:
        genres_str: Genre string, e.g. "Action|Comedy|Drama".

    Returns:
        List of genre strings, e.g. ["Action", "Comedy", "Drama"].
        Returns an empty list for "(no genres listed)".
    """
    if not genres_str or genres_str.strip() == "(no genres listed)":
        return []
    return [g.strip() for g in genres_str.split("|") if g.strip()]


def genres_to_string(genres: List[str]) -> str:
    """
    Convert a list of genres to a space-separated string for TF-IDF vectorisation.

    Multi-word genres like "Sci-Fi" become "SciFi" to prevent tokenisation splits.

    Args:
        genres: List of genre names.

    Returns:
        Space-separated, hyphen-free genre string for TF-IDF.
    """
    return " ".join(g.replace("-", "").replace(" ", "") for g in genres)


# =============================================================================
# Model Artifact Helpers
# =============================================================================


def get_model_path(model_name: str) -> Path:
    """
    Return the expected artifact path for a trained model file.

    Args:
        model_name: Identifier for the model (e.g. "svd_recommender").

    Returns:
        Absolute Path to the joblib file.
    """
    cfg = load_config()
    return resolve_path(cfg.paths.models_dir) / f"{model_name}.joblib"


def ensure_dir(path: Path) -> Path:
    """
    Create a directory and all parent directories if they don't exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path (allows chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Rating Normalisation
# =============================================================================


def normalise_rating(rating: float, min_r: float = 0.5, max_r: float = 5.0) -> float:
    """
    Normalise a rating to the [0, 1] range using min-max scaling.

    Args:
        rating: Raw rating value.
        min_r: Minimum possible rating value.
        max_r: Maximum possible rating value.

    Returns:
        Normalised float in [0, 1].
    """
    if max_r == min_r:
        return 0.5
    return (rating - min_r) / (max_r - min_r)


def clip_rating(rating: float, min_r: float = 0.5, max_r: float = 5.0) -> float:
    """
    Clip a predicted rating to the valid rating range.

    Args:
        rating: Predicted rating (may be out of range).
        min_r: Minimum valid rating.
        max_r: Maximum valid rating.

    Returns:
        Clipped rating within [min_r, max_r].
    """
    return max(min_r, min(max_r, rating))


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_recommendation_result(
    movie_id: int,
    title: str,
    score: float,
    genres: List[str],
    year: Optional[int],
    explanation: str,
    poster_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standardised recommendation result dictionary.

    This is the canonical shape returned by all recommender models and
    consumed by both the API and the Streamlit frontend.

    Args:
        movie_id: Internal MovieLens movie ID.
        title: Clean movie title (without year suffix).
        score: Recommendation score (higher = more relevant).
        genres: List of genre strings.
        year: Release year or None.
        explanation: Human-readable explanation string.
        poster_url: TMDB poster image URL or None.

    Returns:
        Dictionary with standardised keys.
    """
    return {
        "movie_id": movie_id,
        "title": title,
        "score": round(float(score), 4),
        "genres": genres,
        "year": year,
        "explanation": explanation,
        "poster_url": poster_url,
    }


def top_n_from_scores(
    scores: Dict[int, float], n: int = 10, ascending: bool = False
) -> List[Tuple[int, float]]:
    """
    Extract the top-N (movie_id, score) pairs from a score dictionary.

    Args:
        scores: Mapping of item_id -> score.
        n: Number of top items to return.
        ascending: If True, return lowest scores (e.g. for distance metrics).

    Returns:
        List of (item_id, score) tuples sorted by score.
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return sorted_items[:n]
