from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictedCategory:
    advert_category: str
    predicted_category: str | None = None
    advert_id: int | None = None

    @property
    def tp(self) -> bool:
        return self.advert_category == self.predicted_category
