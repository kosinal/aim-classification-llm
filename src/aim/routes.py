"""API routes for flagging classifier."""

from fastapi import APIRouter, HTTPException, Request

from aim.predictor import EmbeddingClassifier
from aim.schemas import AssessRequest, AssessResponse


router = APIRouter(prefix="/api", tags=["flagging"])


@router.post("/project/{project_id}/assess", response_model=AssessResponse)
async def assess_content(
    project_id: int, request_data: AssessRequest, request: Request
) -> AssessResponse:
    """
    Assess content and determine if it should be recommended.

    Args:
        project_id: Project identifier (integer) from URL path
        request_data: Assessment request containing author, title, and summary
        request: FastAPI request object for accessing app state

    Returns:
        Assessment response with recommendation decision, score, and reasoning

    Raises:
        HTTPException: If classifier is not loaded or processing fails
    """
    # Get classifier from app state
    classifier: EmbeddingClassifier | None = request.app.state.classifier

    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded yet")

    # Convert project_id to project string format (e.g., "project_1")
    project_id_str = f"project_{project_id}"

    try:
        # Get prediction from XGBoost embedding classifier
        result = classifier.predict(
            project_id=project_id_str,
            author=request_data.author,
            title=request_data.title,
            summary=request_data.summary,
        )

        return AssessResponse(
            recommend=result["recommend"],
            recommendation_score=result["recommendation_score"],
            project_id=project_id_str,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
