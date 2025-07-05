from typing import Optional

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.contracts.use_case_contract import UseCaseContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.helpers.prompt_helper import format_prompt, parse_prediction_and_confidence
from app.core.prompts.category_prediction_prompt import category_prediction_prompt


class ExperimentOneUseCase(UseCaseContract):
    def __init__(self, llm: LLMClientContract, categories: list[Category]):
        self.llm = llm
        self.category_titles = self._get_category_titles(categories)

    def run(self, advert: Advert) -> Optional[AdvertCategoryPrediction]:
        prompt = format_prompt(
            category_prediction_prompt,
            advert_title=advert.advert_title,
            advert_text=advert.advert_text,
            category_titles=self.category_titles
        )

        model_result = self.llm.generate(prompt)

        predicted_category, confidence = parse_prediction_and_confidence(
            model_result
        )

        if predicted_category:
            return AdvertCategoryPrediction(
                advert_category=advert.category_title,
                predicted_category=predicted_category,
                confidence=confidence
            )

        return None

    def _get_category_titles(self, categories: list[Category]) -> str:
        return ", ".join(category.title for category in categories)
