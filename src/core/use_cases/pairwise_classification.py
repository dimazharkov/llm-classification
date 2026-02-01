import time

from src.core.contracts.category_diff_repository import CategoryDiffRepository
from src.core.contracts.llm_runner import LLMRunner
from src.core.domain.advert import Advert
from src.core.policies.evaluators.pairwise_evaluator import PairwiseEvaluator
from src.core.policies.prompt_context_builders import advert_to_prompt_ctx, category_to_prompt_ctx
from src.core.types.category_diff import CategoryDiff
from src.core.types.category_prediction import PredictedCategory


class PairwiseClassificationUseCase:
    def __init__(self, llm_runner: LLMRunner, pairwise_evaluator: PairwiseEvaluator) -> None:
        self.llm_runner = llm_runner
        self.pairwise_evaluator = pairwise_evaluator

    def run(self, advert: Advert, category_diff_repo: CategoryDiffRepository) -> PredictedCategory | None:
        if not category_diff_repo.all():
            raise ValueError("category_pair_diff is empty")

        self.pairwise_evaluator.init(category_diff_repo.all_titles())

        for _, category_diff in category_diff_repo.all():
            predicted_category = self._predict_category(advert, category_diff)

            self.pairwise_evaluator.add(
                category_diff.category1.title, category_diff.category2.title, predicted_category
            )

        category, score = self.pairwise_evaluator.best()

        if category:
            return PredictedCategory(
                advert_id=advert.advert_id,
                advert_category=advert.category_title,
                predicted_category=category,
            )

        return None

    def _predict_category(self, advert: Advert, category_diff: CategoryDiff) -> str:
        context = {
            "advert": advert_to_prompt_ctx(advert),
            "category1": category_to_prompt_ctx(category_diff.category1),
            "category2": category_to_prompt_ctx(category_diff.category2),
            "difference": category_diff.difference,
        }

        return self.llm_runner.run("category_pair_prediction", context)
