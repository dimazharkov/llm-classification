from typing import Protocol

from core.types.category_prediction import PredictedCategory
from core.types.classification_metrics import ClassificationMetrics


class ExperimentEvaluator(Protocol):
    def evaluate(self, data_list: list[PredictedCategory]) -> ClassificationMetrics: ...