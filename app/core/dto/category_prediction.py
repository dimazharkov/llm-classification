from typing import Optional

from pydantic import BaseModel, computed_field, field_validator


class AdvertCategoryPrediction(BaseModel):
    advert_id: Optional[int] = None
    advert_category: str
    predicted_category: Optional[str] = None
    # confidence: Optional[float] = None

    # @field_validator("confidence")
    # def round_confidence(cls, v):
    #     if v is None:
    #         return None
    #     return round(v, 2)

    # @property
    @computed_field
    def tp(self) -> bool:
        return self.advert_category == self.predicted_category
