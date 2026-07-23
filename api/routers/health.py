"""
API Health Check Router.

Provides an operational health status endpoint used by deployment platforms
(Render, Railway, Kubernetes) to monitor the API's readiness.
"""

from fastapi import APIRouter, Depends, status

from api.schemas import HealthResponse
from api.dependencies import ModelContainer, get_model_container

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API operational health status",
    description="Returns the readiness state of the API and list of trained models currently cached in memory.",
)
def check_health(
    container: ModelContainer = Depends(get_model_container)
) -> HealthResponse:
    """
    Query operational status of the service.

    Returns HTTP 200 OK with details on loaded models.
    """
    loaded_models = container.get_loaded_model_names()
    
    # Determine overall status
    if not container.dataset:
        state = "error: dataset missing"
    elif len(loaded_models) == 0:
        state = "degraded: no models loaded"
    elif len(loaded_models) < 6:
        state = "partially healthy: some models missing"
    else:
        state = "healthy"

    return HealthResponse(
        status=state,
        version=container.cfg.project.version,
        models_loaded=loaded_models,
    )
