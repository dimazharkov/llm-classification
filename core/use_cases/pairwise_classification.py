import time

from app.resources.prompt_strategies.category_pair_prediction_prompt import category_pair_prediction_prompt
from core.contracts.llm_client import LLMClient
from core.domain.advert import Advert
from core.policies.prompt_helper import format_prompt, parse_prediction
from core.types.category_diff import CategoryDiff
from core.types.category_prediction import PredictedCategory
from core.types.prediction_confidence import PredictionConfidence
from infra.evaluators.pairwise_evaluator import PairwiseEvaluator
from infra.repositories.category_pair_diff_repository import CategoryPairDiffRepository


class PairwiseClassificationUseCase:
    def __init__(self, llm: LLMClient, category_evaluator: PairwiseEvaluator):
        self.llm = llm
        self.category_evaluator = category_evaluator

    def run(
        self,
        advert: Advert,
        category_pair_diff: CategoryPairDiffRepository,
        rate_limit: float = 0.5,
    ) -> PredictedCategory | None:
        self.category_evaluator.init(category_pair_diff.all_titles())

        if not category_pair_diff.all():
            print("Achtung! category_pair_diff is empty")
            return None

        for _, category_diff in category_pair_diff.all():
            prediction_confidence = self._predict_category(advert, category_diff)

            if prediction_confidence:
                self.category_evaluator.add(
                    category_diff.category1.title,
                    category_diff.category2.title,
                    prediction_confidence.prediction,
                )

            time.sleep(rate_limit)

        category, score = self.category_evaluator.best()

        if category:
            return PredictedCategory(
                advert_category=advert.category_title,
                predicted_category=category,
            )

        return None

    def _predict_category(self, advert: Advert, category_diff: CategoryDiff) -> PredictionConfidence | None:
        category1_keywords = ", ".join(category_diff.category1.tf_idf) if category_diff.category1.tf_idf else ""
        category2_keywords = ", ".join(category_diff.category2.tf_idf) if category_diff.category2.tf_idf else ""

        prompt = format_prompt(
            category_pair_prediction_prompt,
            advert=advert,
            category1=category_diff.category1,
            category2=category_diff.category2,
            category1_keywords=category1_keywords,
            category2_keywords=category2_keywords,
            difference=category_diff.difference,
        )

        model_result = self.llm.generate(prompt)

        predicted_category = parse_prediction(model_result)

        if predicted_category:
            return PredictionConfidence(prediction=predicted_category)

        return None
