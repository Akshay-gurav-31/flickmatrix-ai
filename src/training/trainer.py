"""
Model Training and Tracking Orchestrator.

Orchestrates the entire machine learning lifecycle:
    1. Data download and ingestion
    2. Feature engineering and split generation
    3. Training of all 5 recommendation algorithms
    4. Evaluation of all algorithms using RMSE, MAE, and Precision@K
    5. Serialization of trained model artifacts to disk for serving
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig

from src.data.downloader import MovieLensDownloader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import RecSysEvaluator
from src.models.popularity_recommender import PopularityRecommender
from src.models.content_based_recommender import ContentBasedRecommender
from src.models.item_cf_recommender import ItemCFRecommender
from src.models.svd_recommender import SVDRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.utils.helpers import ensure_dir, get_model_path, load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommenderTrainer:
    """
    Orchestrator class to run data prep, model training, evaluation, and serialization.

    Attributes:
        cfg: Master configuration OmegaConf DictConfig.
        data_dir: Path to data directory.
        models_dir: Path to save serialized models.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the training orchestrator."""
        self.cfg: DictConfig = load_config(config_path)
        self.data_dir: Path = resolve_path(self.cfg.paths.data_dir)
        self.models_dir: Path = resolve_path(self.cfg.paths.models_dir)

    def run_pipeline(self, force_prep: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Run the end-to-end recommendation training and serialization pipeline.

        Args:
            force_prep: If True, forces reprocessing of the data splits.

        Returns:
            Dictionary mapping model names to evaluation metric dictionaries.
        """
        ensure_dir(self.models_dir)

        logger.info("Starting End-to-End Recommendation Training Pipeline...")

        # ── Step 1: Preprocess Data ────────────────────────────────────────
        preprocessor = DataPreprocessor()
        data = preprocessor.run(force_reprocess=force_prep)

        # ── Step 2: Initialize Evaluator ───────────────────────────────────
        evaluator = RecSysEvaluator(data)

        # ── Step 3: Initialize Models (excluding User-CF as requested) ─────
        models = {
            "popularity": PopularityRecommender(),
            "content_based": ContentBasedRecommender(),
            "item_cf": ItemCFRecommender(),
            "svd": SVDRecommender(),
            "hybrid": HybridRecommender(),
        }

        evaluation_results: Dict[str, Dict[str, float]] = {}

        # ── Step 4: Train, Evaluate and Save each model ────────────────────
        for key, model in models.items():
            logger.info(f"Training: {model.name}...")
            
            # Fit model
            model.fit(data)

            # Evaluate model
            results = evaluator.evaluate_model(model)
            evaluation_results[key] = results

            # Serialize model to disk using canonical key name
            save_path = get_model_path(key)
            model.save(str(save_path))
            logger.info(f"Successfully trained and saved: {model.name}")

        # ── Step 5: Print Summary Comparison Table ─────────────────────────
        self._print_comparison_summary(evaluation_results)

        return evaluation_results

    def _print_comparison_summary(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a formatted Markdown table comparing all model metrics in console.

        Args:
            results: Nested dictionary of model evaluation scores.
        """
        logger.info("Pipeline Complete! Model Performance Comparison:")
        
        # Columns to display in summary
        cols = ["rmse", "mae", "precision_at_10"]
        
        # Compile dataframe
        summary_rows = []
        for model_key, metrics in results.items():
            row = {"Model": model_key}
            for col in cols:
                row[col.upper()] = metrics.get(col, 0.0)
            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)
        # Format metrics
        for col in cols:
            col_upper = col.upper()
            df[col_upper] = df[col_upper].apply(lambda x: f"{x:.4f}")

        # Print using pandas string representation
        logger.info(f"\n{df.to_string(index=False)}")
        logger.info("──────────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    trainer = RecommenderTrainer()
    trainer.run_pipeline()
