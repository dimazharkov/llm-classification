import itertools
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from app.evaluators.base import BaseEvaluator
from app.helpers.prompt_helper import category_selection_prompt

logger = logging.getLogger(__name__)


class PairwiseEvaluator(BaseEvaluator):
    def __init__(self, model, category_params_repository, category_ids: list):
        super().__init__(model)
        self.category_params_repository = category_params_repository
        self.category_ids = category_ids

    def predict(self, item: str) -> Tuple[str, float]:
        category_scores, win_count = self.predict_pairwise(item)
        max_score = max(category_scores.values(), default=0)
        top_cats = [cat for cat, score in category_scores.items() if score == max_score]

        if len(top_cats) > 1:
            confidence = self._resolve_tie(top_cats, category_scores)
            final_category = top_cats[np.argmax(confidence)]
            return final_category, round(float(confidence[np.argmax(confidence)]), 2)
        else:
            final_category = top_cats[0]
            count = win_count.get(final_category, 1)
            confidence = category_scores[final_category] / count
            return final_category, round(confidence, 2)

    def predict_pairwise(self, ad_text: str) -> Tuple[Dict[str, float], Dict[str, int]]:
        category_scores = defaultdict(float)
        win_counts = defaultdict(int)

        for cat1_id, cat2_id in itertools.combinations(self.category_ids, 2):
            category1 = self.get_category_params(cat1_id)
            category2 = self.get_category_params(cat2_id)
            prompt = category_selection_prompt.format(
                ad_text=ad_text,
                category1=category1,
                category2=category2
            )
            print(f"pairwise: [{cat1_id}, {cat2_id}]")
            model_result = self.model.predict(prompt)
            predicted_title, confidence = self.process_model_result(model_result)

            if predicted_title is None:
                print(f"pairwise error!")
            else:
                category_scores[predicted_title] += confidence
                win_counts[predicted_title] += 1
                logger.info(
                    f"pairwise pred: {predicted_title}, conf: {confidence}"
                )

            time.sleep(self.request_timeout)

        return category_scores, win_counts

    def get_category_params(self, category_id: str) -> Dict[str, str]:
        category_params = self.category_params_repository.get_params(category_id)
        for param in ["keywords", "properties", "context"]:
            category_params[param] = ", ".join(category_params.get(param, []))
        return category_params

    def _resolve_tie(self, top_categories: List[str], scores: Dict[str, float]) -> np.ndarray:
        cat_values = np.array([scores[cat] for cat in top_categories])
        probs = np.exp(cat_values) / np.exp(cat_values).sum()
        return probs