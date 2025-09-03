from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    f1: float
    precision: float
    recall: float