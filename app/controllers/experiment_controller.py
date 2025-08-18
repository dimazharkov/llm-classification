import random
import time
from typing import Any

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
    ) -> None:
        self.advert_repository = advert_repository
        self.evaluator = evaluator
        self.saver = saver

    def run(self, use_case: ExperimentContract, num_cases: int = 30, rate_limit: float = 0.5):
        self._execute(use_case, self.advert_repository.get(), num_cases=num_cases, rate_limit=rate_limit)

    def _execute(
        self,
        use_case: ExperimentContract,
        adverts: list[Advert],
        num_cases: int | None = 30,
        rate_limit: float = 0.5,
    ):
        processed: list[Any] = []
        processed_count = 0

        random.shuffle(adverts)

        for i, advert in enumerate(adverts, start=1):
            if advert.advert_id in [
                111, 162, 154, 112, 236, 110, 218, 319, 189, 313, 248, 173, 34, 114, 54, 230, 97, 290, 67, 38, 62, 77,
                87, 11, 165, 72, 27, 225, 143, 350, 133, 5, 3, 269, 264, 252, 289, 243, 229, 200, 151, 53
            ]:
                continue

            print("START")
            advert_category_prediction = use_case.run(advert)

            if advert_category_prediction:
                advert_category_prediction.advert_id = advert.advert_id
                processed.append(advert_category_prediction)
                processed_count += 1

            if processed_count % 10 == 0:
                self.saver.save_list(processed)

            if isinstance(num_cases, int) and processed_count >= num_cases:
                break

            percent = (i / len(adverts)) * 100
            print(f"[{i}/{len(adverts)}] Обработано: {percent:.1f}%")

            time.sleep(rate_limit)

        classification_metrics = self.evaluator.calculate(processed)
        processed.append(classification_metrics)

        self.saver.save_list(processed)


# if advert.advert_id in [
#     111, 162, 154, 112, 236, 110, 218, 319, 189, 313, 248, 173, 34, 114, 54, 230, 97, 290, 67, 38, 62, 77,
#     87, 11, 165, 72, 27, 225, 143, 350, 133, 5, 3, 269, 264, 252, 289, 243, 229, 200, 151, 53
# ]:
#     continue
