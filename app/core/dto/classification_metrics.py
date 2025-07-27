from pydantic import BaseModel, field_validator


class ClassificationMetrics(BaseModel):
    f1: float
    precision: float
    recall: float

    @classmethod
    @field_validator("f1", "precision", "recall")
    def round_metrics(cls, value: float) -> float:
        return round(value, 2)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        return {
            "f1": round(data["f1"], 2),
            "precision": round(data["precision"], 2),
            "recall": round(data["recall"], 2),
        }
