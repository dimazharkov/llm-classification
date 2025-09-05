from typing import Protocol

from core.domain.advert import Advert
from core.types.category_prediction import PredictedCategory


class UseCase(Protocol):
    def run(self, advert: Advert) -> PredictedCategory | None: ...
