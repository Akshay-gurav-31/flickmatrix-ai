"""
API Recommendation Router.

Exposes endpoints to retrieve personalized movie recommendations for users
and contextualized recommendations for specific movies.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
)
from api.dependencies import ModelContainer, get_model_container
from src.utils.exceptions import UserNotFoundError, MovieNotFoundError
from src.utils.logger import get_logger

router = APIRouter(prefix="/recommend", tags=["Recommendations"])
logger = get_logger(__name__)


@router.post(
    "/user",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get personalized movie recommendations for a user",
    description="Calculates and returns top-N movie recommendations using the requested model.",
)
def get_user_recommendations(
    request: RecommendationRequest,
    container: ModelContainer = Depends(get_model_container),
) -> RecommendationResponse:
    """
    Retrieve recommendations for a user.

    Raises:
        HTTPException 404: If the user ID is invalid/not found.
        HTTPException 503: If the selected model is not loaded.
    """
    model_name = request.model.lower()
    
    # Verify model is loaded
    try:
        model = container.get_model(model_name)
    except Exception as exc:
        logger.error(f"Failed to fetch model {model_name} from container: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommender model '{model_name}' is not loaded on server.",
        )

    # Compute recommendations
    try:
        recs_list = model.recommend(
            user_id=request.user_id,
            n=request.n,
            exclude_seen=request.exclude_seen,
        )
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error(f"Error calculating recommendations for user {request.user_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating recommendations.",
        )

    # Format output items matching RecommendationItem schema
    items = []
    for rec in recs_list:
        items.append(
            RecommendationItem(
                movie_id=rec["movie_id"],
                title=rec["title"],
                score=rec["score"],
                genres=rec["genres"],
                year=rec["year"],
                explanation=rec["explanation"],
                poster_url=rec.get("poster_url"),
            )
        )

    return RecommendationResponse(
        user_id=request.user_id,
        model=model.name,
        recommendations=items,
        count=len(items),
    )


# Standard request schema for POST /recommend/movie
from pydantic import BaseModel, Field

class MovieRecommendationRequest(BaseModel):
    movie_id: int = Field(..., description="Target movieId to recommend context for.")
    n: int = Field(default=10, ge=1, le=50, description="Number of items to return.")
    model: str = Field(default="hybrid", description="Recommender model to use.")


@router.post(
    "/movie",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get contextual recommendations related to a movie",
    description="Returns top-N related movie recommendations based on a source movie.",
)
def get_movie_context_recommendations(
    request: MovieRecommendationRequest,
    container: ModelContainer = Depends(get_model_container),
) -> RecommendationResponse:
    """
    Retrieve related movies. This behaves similarly to POST /similar
    but returns context recommendations formatted under RecommendationResponse.
    """
    model_name = request.model.lower()
    
    try:
        model = container.get_model(model_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommender model '{model_name}' is not loaded.",
        )

    try:
        recs_list = model.similar_movies(movie_id=request.movie_id, n=request.n)
    except MovieNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error(f"Error fetching similar movies for movie {request.movie_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating contextual recommendations.",
        )

    items = []
    for rec in recs_list:
        items.append(
            RecommendationItem(
                movie_id=rec["movie_id"],
                title=rec["title"],
                score=rec["score"],
                genres=rec["genres"],
                year=rec["year"],
                explanation=rec["explanation"],
                poster_url=rec.get("poster_url"),
            )
        )

    # We map movie_id to user_id key to reuse response schemas
    return RecommendationResponse(
        user_id=request.movie_id,  # contextual indicator
        model=model.name,
        recommendations=items,
        count=len(items),
    )
