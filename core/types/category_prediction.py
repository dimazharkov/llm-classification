from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class PredictedCategory:
    advert_category: str
    predicted_category: Optional[str] = None
    advert_id: Optional[int] = None

    @property
    def tp(self) -> bool:
        return self.advert_category == self.predicted_category
