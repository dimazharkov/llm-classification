from typing import Protocol, Optional

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.dto.category_prediction import AdvertCategoryPrediction


class ExperimentContract(Protocol):
    def __init__(self, llm: LLMClientContract, categories: list[Category]):
        ...

    def run(self, advert: Advert) -> Optional[AdvertCategoryPrediction]:
        ...