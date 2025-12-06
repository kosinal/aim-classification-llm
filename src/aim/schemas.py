"""Pydantic schemas for flagging classifier API."""

from pydantic import BaseModel, Field


class AssessRequest(BaseModel):
    """Request schema for content assessment."""

    summary: str = Field(
        ...,
        description="The summary of the content to evaluate",
        min_length=1,
        json_schema_extra={"example": "Budget concerns regarding the new marketing initiative."},
    )
    author: str = Field(
        ...,
        description="Name of the author of the article",
        min_length=1,
        json_schema_extra={"example": "J.R.R. Tolkien"},
    )
    title: str = Field(
        ...,
        description="Original title of the article",
        min_length=1,
        json_schema_extra={"example": "Budget concerns regarding the new marketing initiative."},
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
    project_id: str = Field(..., description="The project ID for this assessment")
