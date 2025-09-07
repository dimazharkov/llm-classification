import time
from typing import Optional

from core.contracts.category_diff_repository import CategoryDiffRepository
from core.contracts.llm_runner import LLMRunner
from core.domain.advert import Advert
from core.policies.prompt_context_builders import category_to_prompt_ctx, advert_to_prompt_ctx
from core.types.category_diff import CategoryDiff
from core.types.category_prediction import PredictedCategory
from core.policies.evaluators.pairwise_evaluator import PairwiseEvaluator


class PairwiseClassificationUseCase:
    def __init__(
            self,
            llm_runner: LLMRunner,
            pairwise_evaluator: PairwiseEvaluator,
            rate_limit: float = 0.5
    ):
        self.llm_runner = llm_runner
        self.pairwise_evaluator = pairwise_evaluator
        self.rate_limit = rate_limit

    def run(self, advert: Advert, category_diff: CategoryDiffRepository) -> Optional[PredictedCategory]:
        if not category_diff.all():
            raise ValueError("category_pair_diff is empty")

        self.pairwise_evaluator.init(category_diff.all_titles())

        for _, category_diff in category_diff.all():
            predicted_category = self._predict_category(advert, category_diff)

            self.pairwise_evaluator.add(
                category_diff.category1.title,
                category_diff.category2.title,
                predicted_category
            )

            time.sleep(self.rate_limit)

        category, score = self.pairwise_evaluator.best()

        if category:
            return PredictedCategory(
                advert_id=advert.advert_id,
                advert_category=advert.category_title,
                predicted_category=category
            )

        return None

    def _predict_category(self, advert: Advert, category_diff: CategoryDiff) -> str:
        context = {
            "advert": advert_to_prompt_ctx(advert),
            "category1": category_to_prompt_ctx(category_diff.category1),
            "category2": category_to_prompt_ctx(category_diff.category2),
            "difference": category_diff.difference
        }

        return self.llm_runner.run("category_pair_prediction", context)
