from app.core.contracts.experiment_contract import ExperimentContract
from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.helpers.prompt_helper import format_prompt, parse_prediction_and_confidence
from app.core.prompts.category_prediction_prompt import category_prediction_prompt
from app.repositories.category_file_repository import CategoryFileRepository


class ExperimentOneUseCase(ExperimentContract):
    def __init__(self, llm: LLMClientContract, category_repo: CategoryFileRepository):
        self.llm = llm
        self.category_titles = self._get_category_titles(category_repo)

    def run(self, advert: Advert) -> AdvertCategoryPrediction | None:
        prompt = format_prompt(
            category_prediction_prompt,
            advert_title=advert.advert_title,
            advert_text=advert.advert_text,
            category_titles=self.category_titles,
        )

        model_result = self.llm.generate(prompt)

        predicted_category, confidence = parse_prediction_and_confidence(model_result)

        if predicted_category:
            return AdvertCategoryPrediction(
                advert_category=advert.category_title,
                predicted_category=predicted_category,
                confidence=confidence,
            )

        return None

    def _get_category_titles(self, category_repo: CategoryFileRepository) -> str:
        return ", ".join(category.title for category in category_repo.get())
