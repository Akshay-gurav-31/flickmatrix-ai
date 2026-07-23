"""
API Similar Movies Router.

Exposes endpoints to find similar movies for a target movie
and search the movie catalog database.
"""

from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import SimilarMovieRequest, SimilarMovieResponse, RecommendationItem
from api.dependencies import ModelContainer, get_model_container
from src.utils.exceptions import MovieNotFoundError
from src.utils.logger import get_logger

router = APIRouter(tags=["Similar Movies"])
logger = get_logger(__name__)


@router.post(
    "/similar",
    response_model=SimilarMovieResponse,
    status_code=status.HTTP_200_OK,
    summary="Get similar movies for a target movie",
    description="Returns top-N similar movies based on content overlap, collaborative rating correlations, or latent space traits.",
)
def get_similar_movies(
    request: SimilarMovieRequest,
    container: ModelContainer = Depends(get_model_container),
) -> SimilarMovieResponse:
    """
    Retrieve similar movies.

    Raises:
        HTTPException 404: If the movie ID is invalid/not found.
        HTTPException 503: If the selected model is not loaded.
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
        sims_list = model.similar_movies(movie_id=request.movie_id, n=request.n)
    except MovieNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error(f"Error calculating similar items for movie {request.movie_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating similarities.",
        )

    items = []
    for rec in sims_list:
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

    return SimilarMovieResponse(
        movie_id=request.movie_id,
        model=model.name,
        similar_movies=items,
        count=len(items),
    )


# ── Search Endpoint (Helper utility for Frontend) ───────────────────────────


@router.get(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Search movie catalog",
    description="Query catalog movies by title substring or filter by genre.",
)
def search_movies(
    query: Optional[str] = Query(None, description="Substring to search in movie titles."),
    genre: Optional[str] = Query(None, description="Genre tag to filter by."),
    limit: int = Query(20, ge=1, le=100, description="Max search results."),
    container: ModelContainer = Depends(get_model_container),
):
    """
    Search endpoint returning simple movie dictionaries.
    """
    if container.dataset is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Movie dataset is not loaded on server.",
        )

    df = container.dataset.movies.copy()

    # Apply filters
    if query:
        df = df[df["title"].str.contains(query, case=False, na=False)]
        
    if genre:
        # Check if genre list contains selected genre
        df = df[df["genre_list"].apply(lambda genres: genre.lower() in [g.lower() for g in genres])]

    # Sort search results by popularity (bayesian average) to return the most relevant titles first
    results_df = df.sort_values(by="bayesian_avg", ascending=False).head(limit)

    results = []
    for _, row in results_df.iterrows():
        results.append(
            {
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "clean_title": row["clean_title"],
                "genres": row["genre_list"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "bayesian_avg": round(float(row["bayesian_avg"]), 2),
                "poster_url": row.get("poster_url"),
            }
        )

    return {"results": results, "count": len(results)}


@router.get(
    "/genres",
    status_code=status.HTTP_200_OK,
    summary="Get all genres list",
    description="Returns list of all unique genres present in the dataset catalog.",
)
def get_genres(container: ModelContainer = Depends(get_model_container)):
    """Return catalog unique genres list."""
    if container.dataset is None:
        return {"genres": []}
    
    genres_series = container.dataset.movies["genre_list"].explode().dropna()
    unique_genres = sorted(list(genres_series.unique()))
    return {"genres": unique_genres}
