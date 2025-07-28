import random
import time
from typing import Any, Optional

from app.core.contracts.experiment_contract import ExperimentContract
from app.core.domain.advert import Advert
from app.infrastructure.evaluators.classification_evaluator import ClassificationEvaluator
from app.infrastructure.persistence.json_saver import JsonSaver
from app.repositories.advert_file_repository import AdvertFileRepository


class ExperimentController:
    def __init__(
        self,
        advert_repository: AdvertFileRepository,
        evaluator: ClassificationEvaluator,
        saver: JsonSaver,
    ):
        self.advert_repository = advert_repository
        self.evaluator = evaluator
        self.saver = saver

    def run(self, use_case: ExperimentContract, num_cases: int = 30, rate_limit: int = 1):
        self._execute(use_case, self.advert_repository.get(), num_cases=None, rate_limit=1)

    def _execute(
        self,
        use_case: ExperimentContract,
        adverts: list[Advert],
        num_cases: Optional[int] = 30,
        rate_limit: int = 1,
    ):
        processed: list[Any] = []
        processed_count = 0

        random.shuffle(adverts)

        for i, advert in enumerate(adverts, start=1):
            # print(f"advert={advert.advert_title}")
            advert_category_prediction = use_case.run(advert)

            if advert_category_prediction:
                print("+")
                processed.append(advert_category_prediction)
                processed_count += 1
            else:
                print("-")

            if processed_count % 10 == 0:
                self.saver.save_list(processed)

            if isinstance(num_cases, int) and processed_count >= num_cases:
                break

            time.sleep(rate_limit)

        classification_metrics = self.evaluator.calculate(processed)
        processed.append(classification_metrics)

        self.saver.save_list(processed)
