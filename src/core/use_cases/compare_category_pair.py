import itertools
import time

from src.core.contracts.category_diff_repository import CategoryDiffRepository
from src.core.contracts.category_pair_repository import CategoryPairRepository
from src.core.contracts.category_repository import CategoryRepository
from src.core.contracts.llm_runner import LLMRunner
from src.core.domain.category import Category
from src.core.policies.prompt_context_builders import category_to_prompt_ctx
from src.core.types.category_diff import CategoryDiff


class CompareCategoryPairUseCase:
    def __init__(
        self,
        llm_runner: LLMRunner,
        category_repo: CategoryRepository,
        category_pair_repo: CategoryPairRepository,
        category_diff_repo: CategoryDiffRepository,
    ) -> None:
        self.llm_runner = llm_runner
        self.category_repo = category_repo
        self.category_pair_repo = category_pair_repo
        self.category_diff_repo = category_diff_repo

    def run(self, category_list: list[Category]) -> CategoryDiffRepository:
        if not category_list:
            raise ValueError("category_list is empty")

        for category1, category2 in itertools.combinations(category_list, 2):
            pair_key = (category1.id, category2.id)
            diff_text = self.category_pair_repo.get(pair_key)

            if not diff_text:
                diff_text = self._prep_categories_diff(category1, category2)
                self.category_pair_repo.add(pair_key, diff_text)

            self.category_diff_repo.add(
                pair_key,
                CategoryDiff(category1=category1, category2=category2, difference=diff_text),
            )

        self.category_pair_repo.save()

        return self.category_diff_repo

    def _prep_categories_diff(self, category1: Category, category2: Category) -> str:
        context = {"category1": category_to_prompt_ctx(category1), "category2": category_to_prompt_ctx(category2)}

        return self.llm_runner.run("category_difference", context)
