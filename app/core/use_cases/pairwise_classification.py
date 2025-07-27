import time

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.dto.category_diff import CategoryDiff
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.dto.prediction_confidence import PredictionConfidence
from app.core.helpers.prompt_helper import format_prompt, parse_prediction_and_confidence
from app.core.prompts.category_pair_prediction_prompt import category_pair_prediction_prompt
from app.infrastructure.evaluators.pairwise_evaluator import PairwiseEvaluator
from app.repositories.category_pair_diff_repository import CategoryPairDiffRepository


class PairwiseClassificationUseCase:
    def __init__(self, llm: LLMClientContract, category_evaluator: PairwiseEvaluator):
        self.llm = llm
        self.category_evaluator = category_evaluator

    def run(
        self,
        advert: Advert,
        category_pair_diff: CategoryPairDiffRepository,
        rate_limit: int = 1,
    ) -> AdvertCategoryPrediction | None:
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
        print(f"predicted category: {category if category else ""}")
        if category:
            return AdvertCategoryPrediction(
                advert_category=advert.category_title,
                predicted_category=category,
                confidence=score,
            )

        return None

    def _predict_category(self, advert: Advert, category_diff: CategoryDiff) -> PredictionConfidence | None:
        prompt = format_prompt(
            category_pair_prediction_prompt,
            advert=advert,
            category1=category_diff.category1,
            category2=category_diff.category2,
            difference=category_diff.difference,
        )

        model_result = self.llm.generate(prompt)

        predicted_category, confidence = parse_prediction_and_confidence(model_result)
        if predicted_category:
            return PredictionConfidence(prediction=predicted_category, confidence=confidence)

        return None

    def _predict_category_backup(self, advert: Advert, category_diff: CategoryDiff) -> PredictionConfidence | None:
        prompt = format_prompt(
            category_pair_prediction_prompt,
            advert=advert,
            category1=category_diff.category1,
            category2=category_diff.category2,
            difference=category_diff.difference,
        )

        model_result = self.llm.generate(prompt)

        predicted_category, confidence = parse_prediction_and_confidence(model_result)
        print(f"predicted_category={predicted_category}, confidence={confidence}")
        if predicted_category and predicted_category != "другое":
            print("ok")
            return PredictionConfidence(prediction=predicted_category, confidence=confidence)

        return None
