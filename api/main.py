"""
FastAPI Application Entry Point.

Sets up middleware, exception handlers, includes modular routers,
and initializes the model cache container on startup.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routers import health, recommend, similar
from api.dependencies import ModelContainer, get_model_container
from src.utils.exceptions import (
    RecommendationSystemError,
    UserNotFoundError,
    MovieNotFoundError,
    ModelLoadError,
)
from src.utils.helpers import load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Load configurations
cfg = load_config()

# Configure global logger for FastAPI
setup_logging(
    log_level=cfg.logging.level,
    log_file=cfg.logging.log_file,
    rotation=cfg.logging.rotation,
    retention=cfg.logging.retention,
    compression=cfg.logging.compression,
)

app = FastAPI(
    title=cfg.api.title,
    description=cfg.api.description,
    version=cfg.api.version,
    docs_url=cfg.api.docs_url,
    redoc_url=cfg.api.redoc_url,
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
# Allows Streamlit or frontend clients on other hosts/ports to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.api.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Lifecycle event ──────────────────────────────────────────────────


@app.on_event("startup")
def startup_event():
    """Initialise and cache all trained models in memory on startup."""
    logger.info("Initializing API application container...")
    container = get_model_container()
    loaded = container.get_loaded_model_names()
    logger.info(f"API loaded successfully with models: {loaded}")


# ── Modular Routers ──────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(similar.router)


# ── Exception Handlers ───────────────────────────────────────────────────────


@app.exception_handler(UserNotFoundError)
def handle_user_not_found(request: Request, exc: UserNotFoundError):
    """Handle UserNotFoundError and return a 404 response."""
    logger.warning(f"UserNotFoundError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "UserNotFound", "message": exc.message},
    )


@app.exception_handler(MovieNotFoundError)
def handle_movie_not_found(request: Request, exc: MovieNotFoundError):
    """Handle MovieNotFoundError and return a 404 response."""
    logger.warning(f"MovieNotFoundError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "MovieNotFound", "message": exc.message},
    )


@app.exception_handler(ModelLoadError)
def handle_model_load_error(request: Request, exc: ModelLoadError):
    """Handle ModelLoadError and return a 503 response."""
    logger.error(f"ModelLoadError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "ModelNotLoaded", "message": exc.message},
    )


@app.exception_handler(RecommendationSystemError)
def handle_general_recsys_error(request: Request, exc: RecommendationSystemError):
    """Handle any generic internal RecommendationSystemError."""
    logger.error(f"RecommendationSystemError: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": exc.__class__.__name__, "message": exc.message},
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root redirect endpoint returning server status."""
    return {
        "title": cfg.api.title,
        "status": "online",
        "docs": cfg.api.docs_url,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
    )
