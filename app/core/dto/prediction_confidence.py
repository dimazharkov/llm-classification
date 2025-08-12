from dataclasses import dataclass


@dataclass(slots=True)
class PredictionConfidence:
    prediction: str
    # confidence: float | None = None
