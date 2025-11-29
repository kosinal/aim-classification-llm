"""API routes for flagging classifier."""

from fastapi import APIRouter, HTTPException, Request

from aim.models import FlagClassifier
from aim.schemas import AssessRequest, AssessResponse


router = APIRouter(prefix="/api", tags=["flagging"])


@router.post("/project/{project_id}/assess", response_model=AssessResponse)
async def assess_content(
    project_id: int, request_data: AssessRequest, request: Request
) -> AssessResponse:
    """
    Assess content and determine if it should be flagged.

    Args:
        project_id: Project identifier (integer) from URL path
        request_data: Assessment request containing summary
        request: FastAPI request object for accessing app state

    Returns:
        Assessment response with flagging decision, risk score, and reasoning

    Raises:
        HTTPException: If models are not loaded, project not found, or processing fails
    """
    # Get models map from app state
    models_map: dict[str, FlagClassifier] | None = request.app.state.models

    if models_map is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Convert project_id to string for map lookup
    project_id_str = str(project_id)

    # Check if model exists for this project
    if project_id_str not in models_map:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for project {project_id}. Available projects: {list(models_map.keys())}",
        )

    model_instance = models_map[project_id_str]

    try:
        pred = model_instance(project_id=project_id_str, summary=request_data.summary)

        # Robust parsing (LLMs sometimes return strings for numbers)
        try:
            score = float(pred.prediction_score)
        except (ValueError, TypeError):
            # Fallback logic if parsing fails
            score = 1.0 if str(pred.prediction).lower() == "positive" else 0.0

        recommend = str(pred.prediction).lower() == "positive"

        return AssessResponse(
            recommend=recommend,
            recommendation_score=score,
            reasoning=pred.reasoning,
            project_id=project_id_str,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
