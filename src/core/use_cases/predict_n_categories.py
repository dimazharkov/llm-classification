from src.core.contracts.category_repository import CategoryRepository
from src.core.contracts.llm_runner import LLMRunner
from src.core.domain.advert import Advert
from src.core.domain.category import Category
from src.core.policies.prompt_context_builders import advert_to_prompt_ctx


class PredictNCategoriesUseCase:
    def __init__(self, llm_runner: LLMRunner, category_repo: CategoryRepository) -> None:
        self.llm_runner = llm_runner
        self.category_repo = category_repo

    def run(self, advert: Advert) -> list[Category]:
        context = {
            "advert": advert_to_prompt_ctx(advert),
            "categories_with_kw": self.category_repo.get_all_with_kw(),
        }

        category_titles_list = self.llm_runner.run("n_category_kw_prediction", context)

        return self.category_repo.get_by_titles(category_titles_list)
