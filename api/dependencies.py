"""
API Dependencies and Model Cache Container.

Defines a thread-safe Singleton ModelContainer that loads all serialised models
into memory at startup and exposes them as FastAPI endpoint dependencies,
preventing disk I/O bottlenecks.
"""

from typing import Any, Dict, List, Optional

from src.models.base_recommender import BaseRecommender
from src.data.preprocessor import DataPreprocessor
from src.utils.exceptions import ModelLoadError
from src.utils.helpers import get_model_path, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelContainer:
    """
    Singleton cache container for trained Recommender models.

    Loads and keeps fitted models in memory to ensure low-latency predictions.
    """

    _instance: Optional["ModelContainer"] = None

    def __new__(cls, *args, **kwargs):
        """Implement Singleton pattern."""
        if not cls._instance:
            cls._instance = super(ModelContainer, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise container. Only executes once due to guard."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.cfg = load_config()
        self.models: Dict[str, BaseRecommender] = {}
        self.dataset: Any = None
        
        # Load all models at startup
        self.load_all_models()

    def load_all_models(self) -> None:
        """
        Load all trained models from joblib files.

        If a model doesn't exist, it logs a warning but does not crash the server,
        allowing the API to start in a partially-functioning state.
        """
        model_names = ["popularity", "content_based", "item_cf", "svd", "hybrid"]

        # First, load the preprocessed dataset
        try:
            logger.info("Loading preprocessed dataset for API reference...")
            preprocessor = DataPreprocessor()
            self.dataset = preprocessor.run()
            logger.info("Dataset successfully loaded.")
        except Exception as e:
            logger.critical(
                f"Failed to load processed dataset. Recommenders will fail to predict. "
                f"Run 'python scripts/train.py' first. Error: {e}"
            )
            return

        # Load each model
        for name in model_names:
            path = get_model_path(name)
            if not path.exists():
                logger.warning(
                    f"Trained model artifact not found for '{name}' at {path.name}. "
                    f"Endpoint using this model will return errors. Run training script."
                )
                continue

            try:
                self.models[name] = BaseRecommender.load(str(path))
                logger.info(f"Successfully loaded '{name}' into API container.")
            except Exception as e:
                logger.error(f"Failed to load model '{name}': {e}", exc_info=True)

    def get_model(self, model_name: str) -> BaseRecommender:
        """
        Retrieve a loaded model instance from the registry.

        Args:
            model_name: Slug name of the model.

        Returns:
            Fitted Recommender instance.

        Raises:
            ModelLoadError: If the model is not loaded.
        """
        model = self.models.get(model_name.lower())
        if not model:
            raise ModelLoadError(
                f"Model '{model_name}' is not loaded in memory.",
                details="Verify the model has been trained and saved in artifacts/models/.",
            )
        return model

    def get_loaded_model_names(self) -> List[str]:
        """Return list of model slugs loaded in the container."""
        return list(self.models.keys())


# ── Global Singleton instance ────────────────────────────────────────────────
# Loaded at module import time
_container_instance = ModelContainer()


# =============================================================================
# FastAPI Dependency Injections
# =============================================================================


def get_model_container() -> ModelContainer:
    """FastAPI Dependency: Returns the global ModelContainer singleton."""
    return _container_instance


def get_model(model_name: str) -> BaseRecommender:
    """
    FastAPI Dependency: Retrieves a specific loaded model by name.
    """
    from fastapi import HTTPException, status
    
    try:
        container = get_model_container()
        return container.get_model(model_name)
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc
