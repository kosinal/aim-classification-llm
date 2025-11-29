"""Pydantic schemas for flagging classifier API."""

from pydantic import BaseModel, Field


class AssessRequest(BaseModel):
    """Request schema for content assessment."""

    project_id: str = Field(
        ..., example="1", description="Project ID (1-4)", min_length=1
    )
    summary: str = Field(
        ...,
        example="Budget concerns regarding the new marketing initiative.",
        description="The summary of the content to evaluate",
        min_length=1,
    )


class AssessResponse(BaseModel):
    """Response schema for content assessment."""

    recommend: bool = Field(..., description="Whether the content should be flagged")
    recommendation_score: float = Field(
        ...,
        description="Recommendation score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        ..., description="Step-by-step analysis of the assessment"
    )
    project_id: str = Field(..., description="The project ID for this assessment")
