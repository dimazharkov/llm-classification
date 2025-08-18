from collections.abc import Iterable

import numpy as np
from scipy.stats import gmean


class PairwiseEvaluator:
    def __init__(self):
        self.labels: list[str] = []
        self.index_map: dict[str, int] = {}
        self.matrix: np.ndarray | None = None
        self.n: int = 0

    def init(self, labels: Iterable[str]):
        self.labels = sorted(labels)
        self.index_map = {label: idx for idx, label in enumerate(self.labels)}
        self.n = len(self.labels)
        self.matrix = np.ones((self.n, self.n))

    def add(self, label1: str, label2: str, winner: str):
        i, j = self.index_map[label1], self.index_map[label2]
        if self.matrix is not None:
            if winner == label1:
                self.matrix[i][j] = 2
                self.matrix[j][i] = 0.5
            elif winner == label2:
                self.matrix[i][j] = 0.5
                self.matrix[j][i] = 2
            else:
                self.matrix[i][j] = 0   # 1
                self.matrix[j][i] = 0   # 1

    def scores(self):
        weights = gmean(self.matrix, axis=1)
        normalized = weights / weights.sum()
        return dict(zip(self.labels, normalized, strict=False))

    def best(self):
        score_dict = self.scores()

        if not score_dict:
            return None, None

        max_score = max(score_dict.values())
        top_labels = [label for label, score in score_dict.items() if score == max_score]

        if len(top_labels) == 1:
            return top_labels[0], max_score

        # Tie-break: количество побед над всеми
        wins = dict.fromkeys(top_labels, 0)
        for label in top_labels:
            i = self.index_map[label]
            for j in range(self.n):
                if i == j:
                    continue
                if self.matrix[i][j] > 1:
                    wins[label] += 1

        max_wins = max(wins.values())
        final_candidates = [label for label, count in wins.items() if count == max_wins]
        winner = sorted(final_candidates)[0]
        return winner, score_dict[winner]
