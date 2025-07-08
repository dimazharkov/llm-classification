from typing import Optional

from app.core.domain.advert import Advert
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from app.core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from app.core.use_cases.predict_five_categories import PredictFiveCategoriesUseCase


class ExperimentThreeUseCase:
    def __init__(
            self,
            predict_five_categories_use_case: PredictFiveCategoriesUseCase,
            compare_category_pair_use_case: CompareCategoryPairUseCase,
            pairwise_classification_use_case: PairwiseClassificationUseCase,
            rate_limit: int = 1,
    ):
        self.predict_five_categories_use_case = predict_five_categories_use_case
        self.compare_category_pair_use_case = compare_category_pair_use_case
        self.pairwise_classification_use_case = pairwise_classification_use_case
        self.rate_limit = rate_limit

    def run(self, advert: Advert) -> Optional[AdvertCategoryPrediction]:
        five_predicted_categories = self.predict_five_categories_use_case.run(
            advert
        )

        category_pair_differences = self.compare_category_pair_use_case.run(
            five_predicted_categories,
            self.rate_limit
        )

        return self.pairwise_classification_use_case.run(
            advert,
            category_pair_differences,
            self.rate_limit
        )
