import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.core.types.category_prediction import PredictedCategory
from src.core.types.classification_metrics import ClassificationMetrics


class ClassificationEvaluator:
    def calculate(self, prediction_list: list[PredictedCategory]) -> ClassificationMetrics:
        y_true, y_pred = zip(*[(r.advert_category, r.predicted_category) for r in prediction_list], strict=False)
        return ClassificationMetrics(
            precision=precision_score(y_true, y_pred, average="micro"),
            recall=recall_score(y_true, y_pred, average="micro"),
            f1=f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true)),
        )
