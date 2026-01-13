from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.infra.storage.file_helper import read_json


def normalize_label(s: str) -> str:
    if s is None:
        return ""

    apostr = {"\u2019": "'", "\u02BC": "'", "\u2032": "'", "’": "'", "ʼ": "'"}
    dashes = {"\u2013": "-", "\u2014": "-", "–": "-", "—": "-"}

    # Unicode нормализация
    s = unicodedata.normalize("NFC", s)
    # унификация апострофов и тире
    for k, v in apostr.items():
        s = s.replace(k, v)
    for k, v in dashes.items():
        s = s.replace(k, v)
    # убираем неразрывные и прочие пробелы → обычный пробел
    s = s.replace("\u00A0", " ")
    # схлопываем подряд и тримим
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def load_results(path: str | Path) -> Tuple[List[str], List[str]]:
    results = read_json(path)
    y_true = [normalize_label(r["advert_category"]) for r in results if "advert_category" in r if "advert_category" in r]
    y_pred = [normalize_label(r["predicted_category"]) for r in results if "advert_category" in r if "advert_category" in r]
    return y_true, y_pred


def analyze_experiment(path: str) -> dict[str, dict]:
    y_true, y_pred = load_results(path)          # списки строк меток
    n_samples = len(y_true)

    # 1) Единый алфавит и индексация 1..N
    labels: List[str] = sorted(set(y_true))
    label_indexes: List[int] = list(range(1, len(labels) + 1))
    lab2idx = {lab: i for i, lab in zip(label_indexes, labels)}  # имя -> индекс (1..N)

    # Для матрицы переводим метки в индексы
    y_true_idx = [lab2idx[t] for t in y_true]
    y_pred_idx = [lab2idx.get(p, None) for p in y_pred]  # если вдруг есть вне-алфавитные
    # Заменим None на 0 и добавим 0 в список labels, чтобы не уронить вычисление (не попадут в 1..N)
    if any(v is None for v in y_pred_idx):
        y_pred_idx = [v if v is not None else 0 for v in y_pred_idx]
        cm_labels_for_sk = [0] + label_indexes
    else:
        cm_labels_for_sk = label_indexes

    # 2) Глобальные метрики
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    err = n_samples - tp_total
    fp_total = err
    fn_total = err

    summary = {
        "n_classes": len(labels),
        "n_samples": n_samples,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "precision": prec_micro,
        "recall": rec_micro,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "micro_f1": f1_micro,
    }

    # 3) Per-class метрики (в порядке labels)
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # 4) Confusion matrix по индексам (1..N)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=cm_labels_for_sk)
    # Если добавляли «0»-й класс для out-of-vocab предсказаний — выкинем его строку/столбец
    if cm.shape[0] == len(label_indexes) + 1:
        cm = cm[1:, 1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm_true = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    tp_vec = np.diag(cm)
    fp_vec = cm.sum(axis=0) - tp_vec
    fn_vec = cm.sum(axis=1) - tp_vec

    per_class = (
        pd.DataFrame({
            "label": labels,                 # имя класса
            "label_idx": label_indexes,      # его индекс 1..N
            "support": support.astype(int),
            "precision": p_cls,
            "recall": r_cls,
            "f1": f1_cls,
            "tp": tp_vec.astype(int),
            "fp": fp_vec.astype(int),
            "fn": fn_vec.astype(int),
        })
        .sort_values("label")                # простая алфавитная сортировка
        .reset_index(drop=True)
    )

    return {
        "summary": summary,
        "per_class": per_class,
        "labels": labels,                    # порядок имён
        "label_indexes": label_indexes,      # 1..N
        "cm": cm,
        "cm_norm_true": cm_norm_true,
    }

def save_experiment_result(name: str, res: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = res["summary"]
    per_class: pd.DataFrame = res["per_class"]
    labels: list[str] = res["labels"]
    label_indexes: list[int] = res["label_indexes"]
    cm: np.ndarray = res["cm"]
    cm_norm_true: np.ndarray = res["cm_norm_true"]  # уже нормированная матрица из анализа, если нужна

    # 1) summary (CSV/JSON)
    summary_csv = out / f"{name}_summary.csv"
    # pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    df_summary_t = pd.DataFrame(summary, index=[0]).T.reset_index()
    df_summary_t.columns = ["metric", "value"]
    df_summary_t.to_csv(summary_csv, index=False)
    summary_json = out / f"{name}_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) per-class (CSV/JSON, n_samples вместо support)
    pc = per_class.sort_values("label").reset_index(drop=True)
    pc_csv = out / f"{name}_per_class.csv"
    pc.to_csv(pc_csv, index=False)

    pc_json = out / f"{name}_per_class.json"
    pc_json.write_text(
        json.dumps(
            [
                {
                    "label": row["label"],
                    "n_samples": int(row["support"]),
                    "f1": float(row["f1"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                }
                for _, row in pc.iterrows()
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) таблица соответствия индексов
    labels_index_csv = out / f"{name}_labels_index.csv"
    pd.DataFrame({"idx": label_indexes, "label": labels}).to_csv(labels_index_csv, index=False)

    # 4) графики Confusion Matrix с индексами на осях
    idx_labels = [str(i) for i in label_indexes]  # 1..N
    # Абсолюты
    fig, ax = plot_confusion_heatmap(cm, idx_labels, title=f"{name} — Confusion (abs)", normalize=None)
    cm_abs_png = out / f"{name}_confusion_abs.png"
    fig.savefig(cm_abs_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Нормировка по истинному классу
    fig, ax = plot_confusion_heatmap(cm, idx_labels, title=f"{name} — Confusion (row-normalized)", normalize="true")
    cm_row_png = out / f"{name}_confusion_row_norm.png"
    fig.savefig(cm_row_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "per_class_csv": pc_csv,
        "per_class_json": pc_json,
        "labels_index_csv": labels_index_csv,
        "cm_abs_png": cm_abs_png,
        "cm_row_norm_png": cm_row_png,
    }


def white_ylorrd():
    base = plt.get_cmap("YlOrRd")
    # берём много точек из YlOrRd
    colors = base(np.linspace(0, 1, 256))
    # заменяем самые низкие на белый (например, первые 30)
    colors[:25, :] = [1, 1, 1, 1]   # RGBA = белый
    return LinearSegmentedColormap.from_list("WhiteYlOrRd", colors)

cmap_wyr = white_ylorrd()

def plot_confusion_heatmap(
        cm: np.ndarray,
        labels: list[str],
        title: str = "Confusion matrix (normalized)",
        normalize: str = "true",
        cmap: str = "hot_r" #"Reds" # "YlOrRd"
):
    """
    normalize: 'true' → нормировка по строкам, 'pred' → по столбцам, None → абсолюты.
    """
    arr = cm.astype(float)
    if normalize == "true":
        denom = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom != 0.0)
    elif normalize == "pred":
        denom = arr.sum(axis=0, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom != 0.0)

    n = arr.shape[0]
    fig_w = min(12, max(6, n * 0.25))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, cmap=cmap_wyr, vmin=0, vmax=arr.max() if normalize is None else 1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # тонкие разделители клеток
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("share" if normalize else "count")
    fig.tight_layout()
    return fig, ax

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _to_json(df: pd.DataFrame) -> str:
    return json.dumps(json.loads(df.to_json(orient="records", force_ascii=False)), ensure_ascii=False, indent=2)

def _plot_macro_weighted_over_time(global_df: pd.DataFrame, out_path: Path) -> Path:
    # Порядок берём как есть (никакой сортировки); шкала F1: 0.5..1.0
    x = np.arange(len(global_df))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, global_df["macro_f1"], marker="o", label="Macro F1")
    ax.plot(x, global_df["weighted_f1"], marker="o", label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(global_df["experiment"], rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Macro / Weighted F1 over experiments")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path

def _plot_delta_bars(per_class_df: pd.DataFrame, col_delta: str, title: str, out_path: Path) -> Path:
    dd = per_class_df[["label", col_delta]].dropna().sort_values(col_delta, ascending=True)
    n = len(dd)
    fig_h = min(12, max(5, 0.25 * n))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(dd["label"], dd[col_delta])
    ax.set_xlabel(col_delta)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path

def _plot_delta_bars_alpha(per_class_df: pd.DataFrame, col_delta: str, title: str, out_path: Path) -> Path:
    dd = per_class_df[["label", col_delta]].dropna().sort_values("label", ascending=True)
    n = len(dd)
    fig_h = min(12, max(5, 0.25 * n))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(dd["label"], dd[col_delta])
    ax.set_xlabel(col_delta)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path

def analyze_global(analyzed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    analyzed: {"Exp1": res1, "Exp2": res2, "Exp3": res3}
    Возвращает словарь с:
      - global_summary (строки Exp1/Exp2/Exp3/Δ Exp2−Exp1/Δ Exp3−Exp1)
      - per_class_summary (label, support, F1 по экспериментам + дельты)
    """
    exp_names = list(analyzed.keys())                 # порядок как в вызове
    base = exp_names[0]
    second = exp_names[1] if len(exp_names) > 1 else None
    third = exp_names[-1]

    # 1) global_summary: сначала строки по экспериментам
    rows = []
    for name in exp_names:
        s = analyzed[name]["summary"]
        rows.append({
            "experiment": name,
            "n_classes": s["n_classes"],
            "n_samples": s["n_samples"],
            "macro_f1": s["macro_f1"],
            "weighted_f1": s["weighted_f1"],
            "precision": s["precision"],   # micro
            "recall": s["recall"],         # micro
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
        })
    global_summary = pd.DataFrame(rows)

    # Добавляем строки-дельты (target − base)
    base_row = global_summary.loc[global_summary["experiment"] == base].iloc[0]
    def _delta_row(target_name: str) -> dict:
        t = global_summary.loc[global_summary["experiment"] == target_name].iloc[0]
        return {
            "experiment": f"Δ {target_name} − {base}",
            "n_classes": np.nan, "n_samples": np.nan,
            "macro_f1": t["macro_f1"] - base_row["macro_f1"],
            "weighted_f1": t["weighted_f1"] - base_row["weighted_f1"],
            "precision": t["precision"] - base_row["precision"],
            "recall": t["recall"] - base_row["recall"],
            "tp": t["tp"] - base_row["tp"],
            "fp": t["fp"] - base_row["fp"],
            "fn": t["fn"] - base_row["fn"],
        }
    if second:
        global_summary = pd.concat([global_summary, pd.DataFrame([_delta_row(second)])], ignore_index=True)
    if third and third != base:
        global_summary = pd.concat([global_summary, pd.DataFrame([_delta_row(third)])], ignore_index=True)

    # 2) per_class_summary: label, support (одна), F1 по экспериментам + дельты
    labels_union = sorted(set().union(*[set(analyzed[n]["per_class"]["label"]) for n in exp_names]))
    pcs = pd.DataFrame({"label": labels_union})

    # support берём из базового эксперимента (он у всех одинаков)
    base_pc = analyzed[base]["per_class"][["label", "support"]].rename(columns={"support": "support"})
    pcs = pcs.merge(base_pc, on="label", how="left")

    # F1 по каждому эксперименту
    for name in exp_names:
        pcs = pcs.merge(
            analyzed[name]["per_class"][["label", "f1"]].rename(columns={"f1": f"{name}_f1"}),
            on="label", how="left"
        )

    # Дельты F1
    if second:
        pcs[f"ΔF1_{second}-{base}"] = pcs[f"{second}_f1"] - pcs[f"{base}_f1"]
    pcs[f"ΔF1_{third}-{base}"] = pcs[f"{third}_f1"] - pcs[f"{base}_f1"]

    per_class_summary = pcs.sort_values("label").reset_index(drop=True)

    return {
        "exp_names": exp_names,
        "base": base,
        "second": second,
        "third": third,
        "global_summary": global_summary,
        "per_class_summary": per_class_summary,
    }

def save_global_results(global_res: Dict[str, Any], out_dir: str | Path) -> Dict[str, Path]:
    out = _ensure_dir(out_dir)

    exp_names = global_res["exp_names"]
    base = global_res["base"]
    second = global_res["second"]
    third = global_res["third"]
    global_summary: pd.DataFrame = global_res["global_summary"]
    per_class_summary: pd.DataFrame = global_res["per_class_summary"]

    # 1) global_summary.(csv|json) — со строками-дельтами
    global_csv = out / "global_summary.csv"
    global_summary.to_csv(global_csv, index=False)
    global_json = out / "global_summary.json"
    global_json.write_text(_to_json(global_summary), encoding="utf-8")

    # 2) per_class_summary.(csv|json) — один support, F1 по эксп., дельты
    pcs_csv = out / "per_class_summary.csv"
    per_class_summary.to_csv(pcs_csv, index=False)
    pcs_json = out / "per_class_summary.json"
    pcs_json.write_text(_to_json(per_class_summary), encoding="utf-8")

    paths = {
        "global_summary_csv": global_csv,
        "global_summary_json": global_json,
        "per_class_summary_csv": pcs_csv,
        "per_class_summary_json": pcs_json,
    }

    # 3) ΔF1 bar charts по категориям
    if second and f"ΔF1_{second}-{base}" in per_class_summary.columns:
        # paths[f"deltas_{second}-vs-{base}.png"] = _plot_delta_bars(
        #     per_class_summary, f"ΔF1_{second}-{base}",
        #     f"ΔF1 ({second} − {base}) by class", out / f"deltas_{second}-vs-{base}.png"
        # )
        paths[f"deltas_{second}-vs-{base}.png"] = _plot_delta_bars_alpha(
            per_class_summary, f"ΔF1_{second}-{base}",
            f"ΔF1 ({second} − {base}) by class", out / f"deltas_{second}-vs-{base}.png"
        )

    if f"ΔF1_{third}-{base}" in per_class_summary.columns:
        # paths[f"deltas_{third}-vs-{base}.png"] = _plot_delta_bars(
        #     per_class_summary, f"ΔF1_{third}-{base}",
        #     f"ΔF1 ({third} − {base}) by class", out / f"deltas_{third}-vs-{base}.png"
        # )
        paths[f"deltas_{third}-vs-{base}.png"] = _plot_delta_bars_alpha(
            per_class_summary, f"ΔF1_{third}-{base}",
            f"ΔF1 ({third} − {base}) by class", out / f"deltas_{third}-vs-{base}.png"
        )

    # 4) Macro/Weighted F1 over time (ось Y: 0.5..1.0)
    # Берём только строки-эксперименты (без дельт) для линии
    mask_exp = ~global_summary["experiment"].str.startswith("Δ ")
    mw_df = global_summary.loc[mask_exp, ["experiment", "macro_f1", "weighted_f1"]]
    paths["macro_weighted_over_time.png"] = _plot_macro_weighted_over_time(
        mw_df, out / "macro_weighted_over_time.png"
    )

    return paths

class ReportService:
    def run(self, exp_files: dict[str, Any], out_dir: str) -> str | Path:
        analyzed: dict[str, dict] = {}

        for name, path in exp_files.items():
            res = analyze_experiment(path)
            save_experiment_result(name, res, out_dir)
            analyzed[name] = res

        global_res = analyze_global(analyzed)
        save_global_results(global_res, out_dir)

        return "path"

