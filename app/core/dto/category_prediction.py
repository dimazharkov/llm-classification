from pydantic import BaseModel, computed_field, field_validator


class AdvertCategoryPrediction(BaseModel):
    advert_category: str
    predicted_category: str | None = None
    confidence: float | None = None

    @field_validator("confidence")
    def round_confidence(cls, v):
        return round(v, 2)

    @property
    @computed_field
    def tp(self) -> bool:
        return self.advert_category == self.predicted_category
