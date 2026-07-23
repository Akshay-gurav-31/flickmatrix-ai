"""
Evaluation Metrics for Recommendation Systems.

Contains functions to calculate:
    1. Rating accuracy: RMSE, MAE
    2. Ranking accuracy: Precision@K

Provides simple metrics for model diagnostics.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.exceptions import EvaluationError
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Accuracy Metrics
# =============================================================================


def compute_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Mean Absolute Error (MAE)."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    return float(mean_absolute_error(y_true, y_pred))


# =============================================================================
# Ranking Metrics
# =============================================================================


def precision_at_k(recommended: List[int], actual_relevant: Set[int], k: int) -> float:
    """
    Calculate Precision@K for a user's recommendation list.

    Precision@K = (Relevant Recommended Items in top-K) / K
    """
    if k <= 0:
        raise EvaluationError("K must be a positive integer", details=f"k={k}")
        
    if not recommended or not actual_relevant:
        return 0.0

    recs_at_k = recommended[:k]
    hits = len([item for item in recs_at_k if item in actual_relevant])
    return hits / k


# =============================================================================
# Unified Evaluator Pipeline
# =============================================================================


class RecSysEvaluator:
    """
    Orchestrates execution of simplified evaluation metrics on test data.
    """

    def __init__(self, data: Any, config_path: Optional[str] = None) -> None:
        """
        Initialise with processed dataset.

        Args:
            data: ProcessedData containing test assets.
        """
        self.cfg = load_config(config_path)
        self.data = data

    def evaluate_model(self, model: Any) -> Dict[str, float]:
        """
        Evaluate rating accuracy and ranking quality metrics for a fitted model.

        Args:
            model: Fitted BaseRecommender model.

        Returns:
            Dictionary containing metrics (RMSE, MAE, Precision@10).
        """
        logger.info(f"[{model.name}] Running simplified evaluation suite...")

        # ── Part 1: Rating Predictions (RMSE, MAE) ─────────────────────────
        test_ratings = self.data.test.copy()
        y_true = []
        y_pred = []
        
        for _, row in test_ratings.iterrows():
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            true_rating = float(row["rating"])
            
            try:
                pred = model.predict(user_id, movie_id)
                y_true.append(true_rating)
                y_pred.append(pred)
            except Exception:
                pass
                
        rmse = compute_rmse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)

        # ── Part 2: Ranking Metrics (Precision@10) ─────────────────────────
        k_values = list(self.cfg.evaluation.k_values)
        relevance_thresh = self.cfg.evaluation.relevance_threshold
        eval_users = self.data.test["userId"].unique()
        
        # Limit evaluation set size for speed (e.g. 100 users) if configured
        n_eval_users = self.cfg.evaluation.n_test_users
        if len(eval_users) > n_eval_users:
            np.random.seed(42)
            eval_users = np.random.choice(eval_users, size=n_eval_users, replace=False)

        precisions = {k: [] for k in k_values}

        for user_id in eval_users:
            user_id = int(user_id)
            user_test = self.data.test[self.data.test["userId"] == user_id]
            relevant_items = set(user_test[user_test["rating"] >= relevance_thresh]["movieId"].tolist())
            
            if not relevant_items:
                continue

            max_k = max(k_values)
            try:
                recs_dict = model.recommend(user_id, n=max_k, exclude_seen=True)
                recs = [r["movie_id"] for r in recs_dict]
            except Exception:
                recs = []

            for k in k_values:
                precisions[k].append(precision_at_k(recs, relevant_items, k))

        # Compile final average scores
        results = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }

        # Add ranking averages
        for k in k_values:
            results[f"precision_at_{k}"] = round(float(np.mean(precisions[k])), 4) if precisions[k] else 0.0

        p_key = f"precision_at_{k_values[0]}" if k_values else "precision_at_10"
        logger.info(
            f"[{model.name}] RMSE: {results['rmse']:.4f} | {p_key}: {results.get(p_key, 0.0):.4f}"
        )
        return results
