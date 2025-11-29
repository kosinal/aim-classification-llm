"""DSPy model definitions for flagging classification."""

import dspy


class FlagAssessor(dspy.Signature):
    """
    Analyze the text and determine if it should be recommended to user based on the project context.
    Output a recommendation score between 0.0 and 1.0, where 1.0 is highly recommendable.
    """
    project_id = dspy.InputField(
        desc="The ID of the project this content belongs to. Relevance context depends on this.")
    summary = dspy.InputField(desc="The summary of the content to evaluate.")

    # We ask for a score to allow threshold tuning for your specific FPR requirements
    reasoning = dspy.OutputField(desc="Step-by-step analysis of why this is relevant or not.")
    prediction_score = dspy.OutputField(
        desc="A float score between 0.0 and 1.0 indicating probability of being recommended.")
    prediction = dspy.OutputField(desc="Binary decision: 'positive' or 'negative'.")


class FlagClassifier(dspy.Module):
    """DSPy module for recommendation classification using Chain of Thought reasoning."""

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.ChainOfThought(FlagAssessor)

    def forward(self, project_id: str, summary: str) -> dspy.Prediction:
        """
        Run the recommendation classification model.

        Args:
            project_id: The ID of the project this content belongs to
            summary: The summary of the content to evaluate

        Returns:
            DSPy prediction with reasoning, risk_score, and prediction fields
        """
        return self.prog(project_id=project_id, summary=summary)
