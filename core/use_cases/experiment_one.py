
from core.contracts.category_repository import CategoryRepository
from core.contracts.llm_runner import LLMRunner
from core.contracts.use_case import UseCase
from core.domain.advert import Advert
from core.policies.prompt_context_builders import advert_to_prompt_ctx
from core.types.category_prediction import PredictedCategory


class ExperimentOneUseCase(UseCase):
    def __init__(self, llm_runner: LLMRunner, category_repo: CategoryRepository):
        self.llm_runner = llm_runner
        self.category_repo = category_repo

    def run(self, advert: Advert) -> PredictedCategory | None:
        context = {
            "advert": advert_to_prompt_ctx(advert),
            "category_titles": self.category_repo.get_titles_str(),
        }

        predicted_category = self.llm_runner.run("category_prediction", context)

        if predicted_category:
            return PredictedCategory(
                advert_id=advert.advert_id, advert_category=advert.category_title, predicted_category=predicted_category,
            )

        return None
