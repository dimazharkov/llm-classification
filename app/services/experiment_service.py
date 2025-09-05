import random
import time
from typing import Any

from core.contracts.advert_repository import AdvertRepository
from core.contracts.experiment_evaluator import ExperimentEvaluator
from core.contracts.experiment_repository import ExperimentRepository
from core.contracts.use_case import UseCase


class ExperimentService:
    def __init__(
        self, advert_repo: AdvertRepository, experiment_repo: ExperimentRepository, evaluator: ExperimentEvaluator,
    ) -> None:
        self.advert_repo = advert_repo
        self.experiment_repo = experiment_repo
        self.evaluator = evaluator

    def run(self, use_case: UseCase, num_cases: int = 30, rate_limit: float = 0.5):
        processed: list[Any] = []
        processed_count = 0

        adverts = self.advert_repo.get_all_filtered()
        random.shuffle(adverts)

        print("start >>>")
        for i, advert in enumerate(adverts, start=1):
            predicted_category = use_case.run(advert)

            if predicted_category:
                processed.append(predicted_category)
                processed_count += 1

            if processed_count % 10 == 0:
                self.experiment_repo.save_list(processed)

            if isinstance(num_cases, int) and processed_count >= num_cases:
                break

            percent = (i / len(adverts)) * 100
            print(f"[{i} of {len(adverts)}] processed: {percent:.1f}%")

            time.sleep(rate_limit)

        classification_metrics = self.evaluator.evaluate(processed)
        processed.append(classification_metrics)

        self.experiment_repo.save_list(processed)
        print("<<< done")
