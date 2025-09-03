from collections import defaultdict

import numpy as np

from core.types.prediction_confidence import PredictionConfidence


class CategoryConfidenceEvaluator:
    def __init__(self):
        self.scores = defaultdict(float)
        self.wins = defaultdict(int)

    def reset(self):
        self.scores = defaultdict(float)
        self.wins = defaultdict(int)

    def add(self, prediction_confidence: PredictionConfidence):
        self.scores[prediction_confidence.prediction] += prediction_confidence.confidence
        self.wins[prediction_confidence.prediction] += 1

    def best(self) -> PredictionConfidence | None:
        max_score = max(self.scores.values(), default=0)
        top = [cat for cat, score in self.scores.items() if score == max_score]

        if not top:
            return None

        if len(top) == 1:
            cat = top[0]
            conf = self.scores[cat] / self.wins.get(cat, 1)
            return PredictionConfidence(prediction=cat, confidence=round(conf, 2))

        cat_values = np.array([self.scores[c] for c in top])
        probs = np.exp(cat_values) / np.exp(cat_values).sum()
        best_cat = top[np.argmax(probs)]

        return PredictionConfidence(prediction=best_cat, confidence=round(float(probs[np.argmax(probs)]), 2))
