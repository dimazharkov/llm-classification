import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# def calc_f1(y_true, y_pred):
#     f1_macro = f1_score(y_true, y_pred, average='macro')
#     f1_micro = f1_score(y_true, y_pred, average='micro')
#     f1_weighted = f1_score(y_true, y_pred, average='weighted')
#
#     print(f'Macro F1 Score: {f1_macro:.2f}')
#     print(f'Micro F1 Score: {f1_micro:.2f}')
#     print(f'Weighted F1 Score: {f1_weighted:.2f}')


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
