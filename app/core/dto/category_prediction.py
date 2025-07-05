from pydantic import BaseModel, field_validator, computed_field


class AdvertCategoryPrediction(BaseModel):
    advert_category: str
    predicted_category: str
    confidence: float

    @field_validator('confidence')
    def round_confidence(cls, v):
        return round(v, 2)

    @property
    @computed_field
    def tp(self) -> bool:
        return self.advert_category == self.predicted_category
