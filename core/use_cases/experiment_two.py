from typing import Optional

from core.contracts.category_repository import CategoryRepository
from core.contracts.llm_runner import LLMRunner
from core.contracts.use_case import UseCase
from core.domain.advert import Advert
from core.types.category_prediction import PredictedCategory


class ExperimentTwoUseCase(UseCase):
    def __init__(self, llm_runner: LLMRunner, category_repo: CategoryRepository):
        self.llm_runner = llm_runner
        self.category_repo = category_repo

    def run(self, advert: Advert) -> Optional[PredictedCategory]:
        context = {
            "advert_title": advert.advert_title,
            "advert_text": advert.advert_text,
            "categories_with_kw": self.category_repo.get_all_with_kw()
        }

        predicted_category = self.llm_runner.run("category_kw_prediction", context)

        if predicted_category:
            return PredictedCategory(
                advert_id=advert.advert_id,
                advert_category=advert.category_title,
                predicted_category=predicted_category
            )

        return None
