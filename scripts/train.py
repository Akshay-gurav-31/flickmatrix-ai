"""
CLI Execution Script to train all Recommender System models.

Enables engineers to trigger model training, evaluation, and experiment tracking
directly from the terminal or CI/CD pipelines.

Usage:
    python scripts/train.py --force-prep
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to python system path to ensure relative imports work
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.trainer import RecommenderTrainer
from src.utils.logger import get_logger, setup_logging
from src.utils.helpers import load_config

logger = get_logger(__name__)


def main() -> None:
    """Parse arguments and execute the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Run FlickMatrix AI Recommendation System training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml file.",
    )
    parser.add_argument(
        "--force-prep",
        action="store_true",
        help="Force download and preprocessing/feature-engineering of raw data.",
    )
    args = parser.parse_args()

    # Load configuration to get logging params
    cfg = load_config(args.config)

    # Initialize logging
    setup_logging(
        log_level=cfg.logging.level,
        log_file=cfg.logging.log_file,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        compression=cfg.logging.compression,
    )

    logger.info("Executing training pipeline via CLI...")

    try:
        trainer = RecommenderTrainer(config_path=args.config)
        trainer.run_pipeline(force_prep=args.force_prep)
        logger.info("CLI execution completed successfully!")
    except Exception as e:
        logger.critical(f"Pipeline crashed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
