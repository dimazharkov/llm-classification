import random
import time
from typing import Dict, Any, Tuple, List

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score, f1_score

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.contracts.use_case_contract import UseCaseContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.dto.classification_metrics import ClassificationMetrics
from app.core.use_cases.experiment_one import ExperimentOneUseCase
from app.core.use_cases.experiment_two import ExperimentTwoUseCase
from app.helpers.os_helper import load_from_disc, save_to_disc


class ExperimentController:
    def __init__(self, llm: LLMClientContract):
        self.llm = llm

    def experiment_one(self, adverts_path: str, categories_path: str, target_path: str, num_cases: int = 30, rate_limit: int = 1):
        adverts, categories = self._load_adverts_and_categories(
            adverts_path, categories_path
        )

        use_case = ExperimentOneUseCase(self.llm, categories)

        self._run_experiment(
            use_case, adverts, target_path, num_cases, rate_limit
        )


    def experiment_two(self, adverts_path: str, categories_path: str, target_path: str, num_cases: int = 30, rate_limit: int = 1) -> None:
        adverts, categories = self._load_adverts_and_categories(
            adverts_path, categories_path
        )

        use_case = ExperimentTwoUseCase(self.llm, categories)

        self._run_experiment(
            use_case, adverts, target_path, num_cases, rate_limit
        )

    def _load_adverts_and_categories(self, adverts_path: str, categories_path: str) -> tuple[list[Advert], list[Category]]:
        raw_adverts = load_from_disc(adverts_path)
        adverts = [Advert.model_validate(advert) for advert in raw_adverts]

        raw_categories = load_from_disc(categories_path)
        categories = [Category.model_validate(category) for category in raw_categories]

        return adverts, categories


    def _run_experiment(self, use_case: UseCaseContract, adverts: list[Advert], target_path: str, num_cases: int = 30, rate_limit: int = 1) -> None:
        processed = []
        processed_count = 0

        random.shuffle(adverts)

        for i, advert in enumerate(adverts, start=1):
            advert_category_prediction = use_case.run(advert)

            if advert_category_prediction is None:
                print("Prediction failed!")
            else:
                print(".")
                processed.append(advert_category_prediction)
                processed_count += 1

            if i % 10 == 0:
                self._save_processed(processed, target_path)

            if isinstance(num_cases, int) and i >= num_cases:
                break

            time.sleep(rate_limit)

        classification_metrics = self._calculate_metrics(processed)
        processed.append(classification_metrics)

        self._save_processed(processed, target_path)


    def _save_processed(self, payload: list[Any], target_path: str) -> None:
        processed = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in payload
        ]
        save_to_disc(processed, target_path)

    def _calculate_metrics(self, prediction_list: list[AdvertCategoryPrediction]) -> ClassificationMetrics:
        y_true, y_pred = zip(*[(r.advert_category, r.predicted_category) for r in prediction_list])

        return ClassificationMetrics(
            precision=precision_score(y_true, y_pred, average="micro"),
            recall=recall_score(y_true, y_pred, average="micro"),
            f1=f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
        )
