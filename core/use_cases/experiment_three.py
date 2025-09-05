import time

from core.domain.advert import Advert
from core.types.category_prediction import PredictedCategory
from core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from core.use_cases.predict_n_categories import PredictNCategoriesUseCase


class ExperimentThreeUseCase:
    def __init__(
        self,
        predict_n_categories: PredictNCategoriesUseCase,
        compare_category_pair: CompareCategoryPairUseCase,
        pairwise_classification: PairwiseClassificationUseCase
    ):
        self.predict_n_categories = predict_n_categories
        self.compare_category_pair = compare_category_pair
        self.pairwise_classification = pairwise_classification

    def run(self, advert: Advert) -> PredictedCategory | None:
        five_predicted_categories = self.predict_n_categories.run(
            advert
        )

        category_pair_differences = self.compare_category_pair.run(
            five_predicted_categories
        )

        return self.pairwise_classification.run(
            advert, category_pair_differences
        )
