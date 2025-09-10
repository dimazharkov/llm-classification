from src.core.domain.advert import Advert
from src.core.types.category_prediction import PredictedCategory
from src.core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from src.core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from src.core.use_cases.predict_n_categories import PredictNCategoriesUseCase


class ExperimentThreeUseCase:
    def __init__(
        self,
        predict_n_categories: PredictNCategoriesUseCase,
        compare_category_pair: CompareCategoryPairUseCase,
        pairwise_classification: PairwiseClassificationUseCase,
    ) -> None:
        self.predict_n_categories = predict_n_categories
        self.compare_category_pair = compare_category_pair
        self.pairwise_classification = pairwise_classification

    def run(self, advert: Advert) -> PredictedCategory | None:
        predicted_categories = self.predict_n_categories.run(advert)

        category_pair_differences = self.compare_category_pair.run(predicted_categories)

        return self.pairwise_classification.run(advert, category_pair_differences)
