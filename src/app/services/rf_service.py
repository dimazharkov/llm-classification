import os
from pathlib import Path
from typing import Any, Counter

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

from src.app.config.config import config
from src.core.contracts.advert_repository import AdvertRepository
from src.infra.repositories.json_file_repository import JsonFileRepository
from src.infra.storage.os_helper import save_to_disc, load_from_disc


class RFService:
    def __init__(self):
        self.embedder_name = 'lang-uk/ukr-paraphrase-multilingual-mpnet-base'

    def build_text(self, advert: dict[str, Any]) -> str:
        title = (advert.get("advert_title", advert.get("title")) or "").strip()
        text = (advert.get("advert_text", advert.get("text")) or "").strip()

        parts = []
        if title:
            parts.append(f"TITLE: {title}")
        if text:
            parts.append(f"TEXT: {text}")

        return "\n".join(parts)


    def train(self, train_path: str, artifacts_path: str,  seed: int = 42) -> None:
        """
        - загружает data/source/train.json
        - считает эмбеддинги
        - обучает XGBoost с val для early stopping
        - сохраняет артефакты в data/artifacts/...
        """
        print("Start RF training...")

        train_data: list[dict[str, Any]] = load_from_disc(train_path)

        os.makedirs(artifacts_path, exist_ok=True)

        if not train_data:
            raise ValueError("train.json пустой или не найден")

        # Тексты и таргеты
        texts = [self.build_text(a) for a in train_data]
        y_raw = np.array([a["category_id"] for a in train_data], dtype=np.int64)

        # Ремаппинг category_id -> [0..K-1]
        classes, y = np.unique(y_raw, return_inverse=True)
        k = len(classes)

        # Эмбеддинги считаем один раз на весь train_data
        emb_path = Path(f"{artifacts_path}/embeddings.npy")
        meta_path = Path(f"{artifacts_path}/embeddings.meta.json")

        meta = {
            "embedder": self.embedder_name,
            "normalize_embeddings": True,
            "n_texts": len(texts),
        }

        x_all = None

        if emb_path.exists() and meta_path.exists():
            try:
                # cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cached_meta = load_from_disc(meta_path)
                if cached_meta == meta:
                    x_all = np.load(emb_path).astype(np.float32)
                    # быстрая проверка формы
                    if x_all.shape[0] != len(texts):
                        x_all = None
            except Exception:
                x_all = None

        if x_all is None:
            embedder = SentenceTransformer(self.embedder_name)
            x_all = embedder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            np.save(emb_path, x_all)
            save_to_disc(meta, meta_path)

        # Split train/val для early stopping (стратифицированно) + обработка редких классов
        # Классы с freq==1 невозможно стратифицированно разделить -> держим их только в train.
        freq = Counter(y.tolist())
        rare_idx = np.array([i for i, cls in enumerate(y) if freq[int(cls)] == 1], dtype=np.int64)
        common_idx = np.array([i for i, cls in enumerate(y) if freq[int(cls)] >= 2], dtype=np.int64)

        if len(common_idx) == 0:
            idx_train = np.arange(len(y), dtype=np.int64)
            idx_val = np.array([], dtype=np.int64)
        else:
            y_common = y[common_idx]
            idx_train_common, idx_val = train_test_split(
                common_idx,
                test_size=0.2,
                random_state=seed,
                stratify=y_common,
            )
            idx_train = np.concatenate([idx_train_common, rare_idx])

        x_train = x_all[idx_train]
        y_train = y[idx_train]

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=os.cpu_count(),
            class_weight="balanced_subsample",
            random_state=seed,
        )
        rf.fit(x_train, y_train)

        if len(idx_val) > 0:
            x_val = x_all[idx_val]
            y_val = y[idx_val]

            val_pred = rf.predict(x_val)

            f1_w = f1_score(y_val, val_pred, average="weighted", zero_division=0)
            f1_m = f1_score(y_val, val_pred, average="macro", zero_division=0)
            report_dict = classification_report(y_val, val_pred, output_dict=True, zero_division=0)

            payload = {
                "split": "val",
                "weighted_f1": float(f1_w),
                "macro_f1": float(f1_m),
                "report": report_dict,
            }
            print("VAL weighted-F1:", f1_w)
            print("VAL macro-F1:", f1_m)
            print(report_dict)

        else:
            train_pred = rf.predict(x_train)
            comment = "VAL split skipped: классы с <2 примерами не позволяют сделать stratified val. Метрики ниже — на TRAIN (будут завышены)."

            f1_w = f1_score(y_train, train_pred, average="weighted", zero_division=0)
            f1_m = f1_score(y_train, train_pred, average="macro", zero_division=0)
            report_dict = classification_report(y_train, train_pred, output_dict=True, zero_division=0)

            payload = {
                "comment": comment,
                "split": "train",
                "weighted_f1": float(f1_w),
                "macro_f1": float(f1_m),
                "report": report_dict,
            }
            print(comment)
            print("TRAIN weighted-F1:", f1_w)
            print("TRAIN macro-F1:", f1_m)
            print(report_dict)

        save_to_disc(payload, f"{artifacts_path}/train_metrics.json")

        # Сохраняем артефакты
        joblib.dump(rf, f"{artifacts_path}/rf.joblib")
        np.save(f"{artifacts_path}/classes.npy", classes)

        cfg = {
            "model_type": "random_forest",
            "embedder": self.embedder_name,
            "normalize_embeddings": True,
            "text_template": "TITLE + TEXT",
        }

        save_to_disc(cfg, f"{artifacts_path}/config.json")


    def predict(self, inf_path: str, artifacts_path: str) -> float:
        """
        - загружает модель и артефакты
        - загружает inf_path
        - предсказывает категории
        - сохраняет {inf_stem}_pred.json рядом с inf_path
        - выводит статистику и метрики ТОЛЬКО для категорий, которые есть в inf.json (y_true)
        Возвращает weighted-F1 (или nan).
        """
        print("Start RF prediction...")

        inf_data: list[dict[str, Any]] = load_from_disc(inf_path)
        if not inf_data:
            raise ValueError("inf.json пустой или не найден")

        cfg = load_from_disc(f"{artifacts_path}/config.json")
        classes = np.load(f"{artifacts_path}/classes.npy")

        # Загружаем RandomForest
        model = joblib.load(f"{artifacts_path}/rf.joblib")

        embedder = SentenceTransformer(cfg["embedder"])

        texts = [self.build_text(a) for a in inf_data]
        x = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
        ).astype(np.float32)

        # RF умеет predict_proba
        proba = model.predict_proba(x)  # (N, K)
        pred_idx = proba.argmax(axis=1).astype(np.int64)
        conf = proba.max(axis=1)  # (N,)
        pred_category_id = classes[pred_idx].astype(np.int64)

        # Сохраняем предсказания рядом с inf.json
        inf_path_p = Path(inf_path)
        out_path = inf_path_p.with_name(inf_path_p.stem + "_pred" + inf_path_p.suffix)

        out_rows = []
        for a, cid, c in zip(inf_data, pred_category_id, conf):
            out_rows.append(
                {
                    "advert_id": a.get("advert_id"),
                    "true_category_id": a.get("category_id"),
                    "pred_category_id": int(cid),
                    "confidence": float(c),
                }
            )
        save_to_disc(out_rows, out_path)
        print(f"Predictions saved to: {out_path}")

        # Если нет ground truth — метрики не считаем
        if any("category_id" not in a for a in inf_data):
            print("Ground truth отсутствует: метрики не считаю.")
            return float("nan")

        y_true_cid = np.array([int(a["category_id"]) for a in inf_data], dtype=np.int64)
        y_pred_cid = pred_category_id

        # Метрики по category_id
        f1_w = f1_score(y_true_cid, y_pred_cid, average="weighted", zero_division=0)
        f1_m = f1_score(y_true_cid, y_pred_cid, average="macro", zero_division=0)
        acc = accuracy_score(y_true_cid, y_pred_cid)

        print(f"INF accuracy: {acc:.3f}")
        print(f"INF weighted-F1: {f1_w:.3f}")
        print(f"INF macro-F1: {f1_m:.3f}")

        # Отчёт: ТОЛЬКО по категориям, которые реально есть в inf.json (в y_true)
        true_labels = np.unique(y_true_cid)
        report_dict = classification_report(
            y_true_cid,
            y_pred_cid,
            labels=true_labels,
            output_dict=True,
            zero_division=0,
        )
        print(report_dict)

        # Доп. диагностика: доля предсказаний вне множества истинных категорий inf.json
        outside = ~np.isin(y_pred_cid, true_labels)
        outside_mean = outside.mean()
        outside_count = int(outside.sum())
        total = int(len(outside))
        outside_rate = float(outside_count / total) if total else 0.0

        print(
            f"Predictions outside inf labels: {outside_mean:.1%} "
            f"({outside_count}/{total})"
        )

        # Статистика по уверенности
        q10, q25, q50, q75, q90 = np.quantile(conf, [0.10, 0.25, 0.50, 0.75, 0.90])
        print(
            "Confidence quantiles:",
            f"p10={q10:.3f} p25={q25:.3f} p50={q50:.3f} p75={q75:.3f} p90={q90:.3f}"
        )

        payload = {
            "accuracy": acc,
            "weighted_f1": float(f1_w),
            "macro_f1": float(f1_m),
            "report": report_dict,
            "conf_quantiles": [q10, q25, q50, q75, q90],
            "outside_inf_labels": {
                "count": outside_count,
                "total": total,
                "rate": outside_rate,  # 0..1
                "mean": outside_mean,
                "percent": outside_rate * 100.0,  # 0..100
            }
        }

        save_to_disc(payload, f"{artifacts_path}/predict_metrics.json")

        return float(f1_w)