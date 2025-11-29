"""API routes for flagging classifier."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from aim.models import FlagClassifier
from aim.schemas import AssessRequest, AssessResponse


router = APIRouter(prefix="/api", tags=["flagging"])


@router.post("/assess", response_model=AssessResponse)
async def assess_content(request_data: AssessRequest, request: Request) -> AssessResponse:
    """
    Assess content and determine if it should be flagged.

    Args:
        request_data: Assessment request containing project_id and summary
        request: FastAPI request object for accessing app state

    Returns:
        Assessment response with flagging decision, risk score, and reasoning

    Raises:
        HTTPException: If model is not loaded or processing fails
    """
    model_instance: FlagClassifier | None = request.app.state.model

    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        pred = model_instance(
            project_id=request_data.project_id, summary=request_data.summary
        )

        # Robust parsing (LLMs sometimes return strings for numbers)
        try:
            score = float(pred.prediction_score)
        except (ValueError, TypeError):
            # Fallback logic if parsing fails
            score = 1.0 if str(pred.prediction).lower() == "true" else 0.0

        recommend = str(pred.prediction).lower() == "positive"

        return AssessResponse(
            recommend=recommend,
            recommendation_score=score,
            reasoning=pred.reasoning,
            project_id=request_data.project_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
