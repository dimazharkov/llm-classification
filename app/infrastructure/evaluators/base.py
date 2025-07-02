import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from app.helpers.model_helper import save_interim_results
from app.helpers.text_helper import clean_text

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    def __init__(self, model):
        self.model = model
        self.request_timeout = 20

    def evaluate(self, items: list, limit: int = 30):
        if not items:
            raise ValueError("No items provided")

        y_true = []
        y_pred = []
        interim_results = []

        processed_count = 0
        i = 0
        logger.info(f"start")
        while processed_count < limit and i < len(items):
            item = items[i]
            i += 1

            predicted_title, confidence = self.predict(item['text'])

            if predicted_title is None:
                logger.warning("Prediction failed!")
            else:
                true_title = clean_text(item["category_title"])
                predicted_title = clean_text(predicted_title)

                y_true.append(true_title)
                y_pred.append(predicted_title)

                self._log_prediction(true_title, predicted_title, confidence)

                interim_results.append({
                    "true": true_title,
                    "pred": predicted_title,
                    "conf": confidence,
                    "TP": true_title == predicted_title,
                })

                if processed_count > 1 and processed_count % 10 == 0:
                    self._save_iterim(interim_results)

                processed_count += 1

            time.sleep(self.request_timeout)

        self._save_iterim(interim_results)
        return y_true, y_pred

    def _log_prediction(self, true_title: str, predicted_title: str, confidence: float):
        logger.info(
            f"true: {true_title}, pred: {predicted_title}, conf: {confidence}, TP: {true_title == predicted_title}"
        )

    def _save_iterim(self, results: list[dict]):
        save_interim_results(results)

    def process_model_result(self, model_result: str) -> tuple[Optional[str], Optional[float]]:
        parts = [x.strip().strip("[]") for x in model_result.split(",")]

        if len(parts) == 2:
            try:
                predicted_title, confidence = parts
                return str(predicted_title), float(confidence)
            except ValueError:
                logger.error(f"model_result: {model_result}")
                logger.error(f"Invalid confidence value!")
        else:
            logger.error(f"Failed to parse model_result: {model_result}")
        return None, None


    @abstractmethod
    def predict(self, item) -> tuple:
        pass
