import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def calc_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def calc_f1(y_true, y_pred) -> float:
    f1: float = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
    return f1


def label_encoder(y_true: list, y_pred: list) -> tuple[list, list]:
    labels = list(set(y_true) | set(y_pred))

    encoder = LabelEncoder()
    encoder.fit(labels)

    y_true_encoded = encoder.transform(y_true)
    y_pred_encoded = encoder.transform(y_pred)

    return y_true_encoded, y_pred_encoded
