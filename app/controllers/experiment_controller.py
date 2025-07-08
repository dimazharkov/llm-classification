import random
import time
from typing import Type

from app.core.contracts.experiment_contract import ExperimentContract
from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.infrastructure.evaluators.classification_evaluator import ClassificationEvaluator
from app.infrastructure.persistence.json_saver import JsonSaver
from app.repositories.advert_file_repository import AdvertFileRepository
from app.repositories.category_file_repository import CategoryFileRepository


class ExperimentController:
    def __init__(
            self,
            advert_repository: AdvertFileRepository,
            evaluator: ClassificationEvaluator | None = None,
            saver: JsonSaver | None = None,
    ):
        self.advert_repository = advert_repository
        self.evaluator = evaluator
        self.saver = saver

    def run(self, use_case: ExperimentContract, num_cases: int = 30, rate_limit: int = 1):
        self._execute(use_case, self.advert_repository.get(), num_cases, rate_limit)

    def _execute(
        self,
        use_case: ExperimentContract,
        adverts: list[Advert],
        num_cases: int = 30,
        rate_limit: int = 1
    ):
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

            if processed_count % 10 == 0:
                self.saver.save_list(processed)

            if isinstance(num_cases, int) and processed_count >= num_cases:
                break

            time.sleep(rate_limit)

        classification_metrics = self.evaluator.calculate(processed)
        processed.append(classification_metrics)

        self.saver.save_list(processed)
