"""
API Request and Response Data Schemas.

Defines Pydantic models for request body validation and response serialization,
ensuring type safety and automatic Swagger/OpenAPI documentation generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================


class RecommendationRequest(BaseModel):
    """Request schema for user-based movie recommendations."""

    user_id: int = Field(
        ...,
        description="The ID of the user requesting recommendations.",
        example=1,
    )
    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return (1 to 50).",
        example=10,
    )
    model: str = Field(
        default="hybrid",
        description="Recommender model to use: 'popularity', 'content_based', 'item_cf', 'svd', 'hybrid'",
        example="hybrid",
    )
    exclude_seen: bool = Field(
        default=True,
        description="If true, filters out movies the user has already rated.",
    )


class SimilarMovieRequest(BaseModel):
    """Request schema for retrieving similar movies."""

    movie_id: int = Field(
        ...,
        description="The target movieId to find similarities for.",
        example=1,
    )
    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of similar movies to return (1 to 50).",
        example=10,
    )
    model: str = Field(
        default="hybrid",
        description="Similarity model to use: 'popularity', 'content_based', 'item_cf', 'svd', 'hybrid'",
        example="hybrid",
    )


# =============================================================================
# Response Schemas
# =============================================================================


class RecommendationItem(BaseModel):
    """Schema representing a single recommended movie element."""

    movie_id: int = Field(..., description="Internal movie ID.")
    title: str = Field(..., description="Movie title (without release year suffix).")
    score: float = Field(..., description="Recommendation score (higher is more relevant).")
    genres: List[str] = Field(default_factory=list, description="List of genres.")
    year: Optional[int] = Field(None, description="Release year.")
    explanation: str = Field(..., description="Human-readable explanation of why this was recommended.")
    poster_url: Optional[str] = Field(None, description="TMDB poster image URL.")


class RecommendationResponse(BaseModel):
    """Response schema containing list of recommended items for a user."""

    user_id: int = Field(..., description="The target user ID.")
    model: str = Field(..., description="The name of the recommender model used.")
    recommendations: List[RecommendationItem] = Field(..., description="Ordered list of recommendations.")
    count: int = Field(..., description="Number of recommendations returned.")


class SimilarMovieResponse(BaseModel):
    """Response schema containing list of similar movies."""

    movie_id: int = Field(..., description="The source movie ID.")
    model: str = Field(..., description="The name of the similarity model used.")
    similar_movies: List[RecommendationItem] = Field(..., description="Ordered list of similar movies.")
    count: int = Field(..., description="Number of similar movies returned.")


class HealthResponse(BaseModel):
    """Response schema containing API status details."""

    status: str = Field(..., description="API operational status (e.g. 'healthy').")
    version: str = Field(..., description="Project application version.")
    models_loaded: List[str] = Field(..., description="List of models successfully loaded in memory.")
