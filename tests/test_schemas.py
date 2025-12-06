"""Tests for Pydantic schemas."""

import pytest

from pydantic import ValidationError

from aim.schemas import AssessRequest, AssessResponse


class TestAssessRequest:
    """Tests for AssessRequest schema."""

    def test_assess_request_valid_data(self):
        """Test AssessRequest with valid data."""
        request = AssessRequest(
            summary="This is a valid summary", author="John Doe", title="Test Title"
        )

        assert request.summary == "This is a valid summary"
        assert request.author == "John Doe"
        assert request.title == "Test Title"

    def test_assess_request_with_long_summary(self):
        """Test AssessRequest accepts long summaries."""
        long_summary = "a" * 10000
        request = AssessRequest(summary=long_summary, author="Author Name", title="Title")

        assert request.summary == long_summary
        assert len(request.summary) == 10000

    def test_assess_request_empty_string_fails(self):
        """Test that empty string fails validation due to min_length."""
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest(summary="", author="Author", title="Title")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "string_too_short"
        assert errors[0]["loc"] == ("summary",)

    def test_assess_request_missing_summary_fails(self):
        """Test that missing summary field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest(author="Author", title="Title")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("summary",) for error in errors)

    def test_assess_request_missing_author_fails(self):
        """Test that missing author field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest(summary="Summary", title="Title")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("author",) for error in errors)

    def test_assess_request_missing_title_fails(self):
        """Test that missing title field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest(summary="Summary", author="Author")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("title",) for error in errors)

    def test_assess_request_missing_all_fields_fails(self):
        """Test that missing all required fields fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessRequest()

        errors = exc_info.value.errors()
        field_names = [error["loc"][0] for error in errors]
        assert "summary" in field_names
        assert "author" in field_names
        assert "title" in field_names

    def test_assess_request_none_summary_fails(self):
        """Test that None summary fails validation."""
        with pytest.raises(ValidationError):
            AssessRequest(summary=None, author="Author", title="Title")

    def test_assess_request_non_string_summary_fails(self):
        """Test that non-string summary fails validation."""
        with pytest.raises(ValidationError):
            AssessRequest(summary=123, author="Author", title="Title")

    def test_assess_request_dict_to_model(self):
        """Test creating AssessRequest from dictionary."""
        data = {"summary": "Test summary from dict", "author": "Dict Author", "title": "Dict Title"}
        request = AssessRequest(**data)

        assert request.summary == "Test summary from dict"
        assert request.author == "Dict Author"
        assert request.title == "Dict Title"

    def test_assess_request_model_dump(self):
        """Test dumping AssessRequest to dictionary."""
        request = AssessRequest(summary="Test summary", author="Test Author", title="Test Title")
        data = request.model_dump()

        assert isinstance(data, dict)
        assert data["summary"] == "Test summary"
        assert data["author"] == "Test Author"
        assert data["title"] == "Test Title"

    def test_assess_request_model_dump_json(self):
        """Test dumping AssessRequest to JSON."""
        request = AssessRequest(summary="Test summary", author="Test Author", title="Test Title")
        json_str = request.model_dump_json()

        assert isinstance(json_str, str)
        assert "Test summary" in json_str
        assert "Test Author" in json_str
        assert "Test Title" in json_str


class TestAssessResponse:
    """Tests for AssessResponse schema."""

    def test_assess_response_valid_data(self):
        """Test AssessResponse with valid data."""
        response = AssessResponse(
            recommend=True,
            recommendation_score=0.85,
            project_id="1",
        )

        assert response.recommend is True
        assert response.recommendation_score == 0.85
        assert response.project_id == "1"

    def test_assess_response_minimum_score(self):
        """Test AssessResponse accepts minimum score of 0.0."""
        response = AssessResponse(
            recommend=False, recommendation_score=0.0, project_id="1"
        )

        assert response.recommendation_score == 0.0

    def test_assess_response_maximum_score(self):
        """Test AssessResponse accepts maximum score of 1.0."""
        response = AssessResponse(
            recommend=True, recommendation_score=1.0, project_id="1"
        )

        assert response.recommendation_score == 1.0

    def test_assess_response_score_below_minimum_fails(self):
        """Test that score below 0.0 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessResponse(
                recommend=False, recommendation_score=-0.1, project_id="1"
            )

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("recommendation_score",) and error["type"] == "greater_than_equal"
            for error in errors
        )

    def test_assess_response_score_above_maximum_fails(self):
        """Test that score above 1.0 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessResponse(
                recommend=True, recommendation_score=1.5, project_id="1"
            )

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("recommendation_score",) and error["type"] == "less_than_equal"
            for error in errors
        )

    def test_assess_response_missing_required_fields(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            AssessResponse(recommend=True)

        errors = exc_info.value.errors()
        field_names = [error["loc"][0] for error in errors]
        assert "recommendation_score" in field_names
        assert "project_id" in field_names

    def test_assess_response_recommend_coerced_to_boolean(self):
        """Test that recommend values are coerced to boolean."""
        # Pydantic v2 coerces truthy/falsy values to boolean
        response = AssessResponse(
            recommend="yes",
            recommendation_score=0.5,
            project_id="1",
        )
        assert isinstance(response.recommend, bool)
        assert response.recommend is True

    def test_assess_response_score_not_float_coerced(self):
        """Test that integer scores are coerced to float."""
        response = AssessResponse(
            recommend=True, recommendation_score=1, project_id="1"
        )

        assert isinstance(response.recommendation_score, float)
        assert response.recommendation_score == 1.0

    def test_assess_response_model_dump(self):
        """Test dumping AssessResponse to dictionary."""
        response = AssessResponse(
            recommend=True, recommendation_score=0.75, project_id="2"
        )
        data = response.model_dump()

        assert isinstance(data, dict)
        assert data["recommend"] is True
        assert data["recommendation_score"] == 0.75
        assert data["project_id"] == "2"

    def test_assess_response_model_dump_json(self):
        """Test dumping AssessResponse to JSON."""
        response = AssessResponse(
            recommend=False, recommendation_score=0.25, project_id="3"
        )
        json_str = response.model_dump_json()

        assert isinstance(json_str, str)
        assert "0.25" in json_str
        assert "project_3" in json_str or "3" in json_str

    def test_assess_response_from_dict(self):
        """Test creating AssessResponse from dictionary."""
        data = {
            "recommend": True,
            "recommendation_score": 0.9,
            "project_id": "5",
        }
        response = AssessResponse(**data)

        assert response.recommend is True
        assert response.recommendation_score == 0.9
        assert response.project_id == "5"

    def test_assess_response_edge_case_scores(self):
        """Test various edge case scores within valid range."""
        test_scores = [0.0, 0.1, 0.5, 0.999, 1.0]

        for score in test_scores:
            response = AssessResponse(
                recommend=score > 0.5,
                recommendation_score=score,
                project_id="1",
            )
            assert response.recommendation_score == score
