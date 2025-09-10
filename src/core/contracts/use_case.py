from typing import Protocol

from src.core.domain.advert import Advert
from src.core.types.category_prediction import PredictedCategory


class UseCase(Protocol):
    def run(self, advert: Advert) -> PredictedCategory | None: ...
