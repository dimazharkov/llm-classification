import time
from collections import defaultdict
from typing import Optional

import numpy as np
from distlib.markers import evaluator

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.dto.category_diff import CategoryDiff
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.dto.prediction_confidence import PredictionConfidence
from app.core.helpers.prompt_helper import format_prompt, parse_prediction_and_confidence
from app.core.prompts.category_pair_prediction_prompt import category_pair_prediction_prompt
from app.infrastructure.evaluators.category_confidence_evaluator import CategoryConfidenceEvaluator
from app.repositories.category_pair_diff_repository import CategoryPairDiffRepository


class PairwiseClassificationUseCase:
    def __init__(self, llm: LLMClientContract, category_evaluator: CategoryConfidenceEvaluator):
        self.llm = llm
        self.category_evaluator = category_evaluator

    def run(self, advert: Advert, category_pair_diff: CategoryPairDiffRepository, rate_limit: int = 1) -> Optional[AdvertCategoryPrediction]:
        self.category_evaluator.reset()
        for _, category_diff in category_pair_diff.all():
            prediction_confidence = self._predict_category(advert, category_diff)
            if prediction_confidence:
                self.category_evaluator.add(prediction_confidence)
            time.sleep(rate_limit)

        prediction_confidence = self.category_evaluator.best()

        if prediction_confidence:
            return AdvertCategoryPrediction(
                advert_category=advert.category_title,
                predicted_category=prediction_confidence.prediction,
                confidence=prediction_confidence.confidence,
            )

        return None

    def _predict_category(self, advert: Advert, category_diff: CategoryDiff) -> Optional[PredictionConfidence]:
        prompt = format_prompt(
            category_pair_prediction_prompt,
            advert=advert,
            category1=category_diff.category1,
            category2=category_diff.category2,
            difference=category_diff.difference
        )

        model_result = self.llm.generate(prompt)

        predicted_category, confidence = parse_prediction_and_confidence(
            model_result
        )
        print(f"predicted_category={predicted_category}, confidence={confidence}")
        if predicted_category and predicted_category != "другое":
            print("ok")
            return PredictionConfidence(
                prediction=predicted_category,
                confidence=confidence
            )

        return None
