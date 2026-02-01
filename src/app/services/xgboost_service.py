import os
from pathlib import Path
from typing import Any, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

from src.app.config.config import config
from src.core.contracts.advert_repository import AdvertRepository
from src.infra.repositories.json_file_repository import JsonFileRepository
from src.infra.storage.os_helper import save_to_disc, load_from_disc


class XGBoostService:
    def __init__(self):
        self.embedder_name = 'lang-uk/ukr-paraphrase-multilingual-mpnet-base'

    def build_text(self, advert: dict[str, Any]) -> str:
        title = (advert.get("advert_title", advert.get("title")) or "").strip()
        text = (advert.get("advert_text", advert.get("text")) or "").strip()
        # summary = (advert.get("advert_summary") or "").strip()

        parts = []
        if title:
            parts.append(f"TITLE: {title}")
        if text:
            parts.append(f"TEXT: {text}")
        # if summary:
        #     parts.append(f"SUMMARY: {summary}")

        return "\n".join(parts)

    def prep(self, raw_path: str, dest_path: str, test_size: float = 0.1, seed: int = 42) -> None:
        """
        - загружает исходный набор данных (JSON-массив объявлений)
        - делит на train / inf (стратифицированно по category_id)
        - сохраняет в data/source/train.json, data/source/inf.json
        """
        raw = load_from_disc(raw_path)

        if not isinstance(raw, list) or not raw:
            raise ValueError("raw_path должен содержать JSON-массив объявлений")

        y = [int(a["category_id"]) for a in raw]
        freq = Counter(y)

        # 1) редкие (1 шт) -> только в train
        rare = [a for a in raw if freq[int(a["category_id"])] == 1]
        common = [a for a in raw if freq[int(a["category_id"])] >= 2]

        if not common:
            # вообще нечего стратифицировать
            save_to_disc(raw, f"{dest_path}/train.json")
            save_to_disc([], f"{dest_path}/inf.json")
            return

        y_common = np.array([int(a["category_id"]) for a in common], dtype=np.int64)

        train_c, inf = train_test_split(
            common,
            test_size=test_size,
            random_state=seed,
            stratify=y_common,
        )

        train = train_c + rare  # добавили редкие в train

        save_to_disc(train, f"{dest_path}/train.json")
        save_to_disc(inf, f"{dest_path}/inf.json")

        print("prep: done")


    def train(self, train_path: str, artifacts_path: str,  seed: int = 42) -> None:
        """
        - загружает data/source/train.json
        - считает эмбеддинги
        - обучает XGBoost с val для early stopping
        - сохраняет артефакты в data/artifacts/...
        """
        train_data: list[dict[str, Any]] = load_from_disc(train_path)

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
            # meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
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
        dtrain = xgb.DMatrix(x_train, label=y_train)

        params = {
            "objective": "multi:softprob",
            "num_class": k,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "eta": 0.15,  # было 0.05
            "max_depth": 4,  # было 6
            "min_child_weight": 5,  # ускоряет и стабилизирует
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "lambda": 1.0,
            "seed": seed,
            "nthread": os.cpu_count(),
        }

        # Если val возможен — используем early stopping; иначе обучаем фиксированное число раундов.
        if len(idx_val) > 0:
            x_val = x_all[idx_val]
            y_val = y[idx_val]
            dval = xgb.DMatrix(x_val, label=y_val)

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1200,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=25,
            )

            val_proba = booster.predict(dval)
            val_pred = val_proba.argmax(axis=1)

            print("VAL weighted-F1:", f1_score(y_val, val_pred, average="weighted"))
            print("VAL macro-F1:", f1_score(y_val, val_pred, average="macro"))
            print(classification_report(y_val, val_pred))
        else:
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=800,
                evals=[(dtrain, "train")],
                verbose_eval=25,
            )

            # Метрики на TRAIN (диагностика; будут завышены, но лучше чем ничего)
            train_proba = booster.predict(dtrain)
            train_pred = train_proba.argmax(axis=1)

            print("VAL split skipped: классы с <2 примерами не позволяют сделать stratified val. Метрики ниже — на TRAIN (будут завышены).")
            print("TRAIN weighted-F1:", f1_score(y_train, train_pred, average="weighted"))
            print("TRAIN macro-F1:", f1_score(y_train, train_pred, average="macro"))
            print(classification_report(y_train, train_pred))

        # Сохраняем артефакты
        booster.save_model(f"{artifacts_path}/xgb.json")
        np.save(f"{artifacts_path}/classes.npy", classes)

        cfg = {
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
        inf_data: list[dict[str, Any]] = load_from_disc(inf_path)
        if not inf_data:
            raise ValueError("inf.json пустой или не найден")

        cfg = load_from_disc(f"{artifacts_path}/config.json")
        classes = np.load(f"{artifacts_path}/classes.npy")

        # Загружаем Booster (если модель обучалась через xgb.train)
        booster = xgb.Booster()
        booster.load_model(f"{artifacts_path}/xgb.json")

        embedder = SentenceTransformer(cfg["embedder"])

        texts = [self.build_text(a) for a in inf_data]
        x = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
        ).astype(np.float32)

        d = xgb.DMatrix(x)
        proba = booster.predict(d)  # (N, K)
        pred_idx = proba.argmax(axis=1)  # (N,)
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