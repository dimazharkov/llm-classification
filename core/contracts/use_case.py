from typing import Protocol, Optional

from core.contracts.llm_client import LLMClient
from core.domain.advert import Advert
from core.domain.category import Category
from core.types.category_prediction import PredictedCategory


class UseCase(Protocol):
    def run(self, advert: Advert) -> Optional[PredictedCategory]: ...
