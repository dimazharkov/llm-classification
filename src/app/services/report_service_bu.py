from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def _load_results(path: str | Path) -> List[Dict[str, Any]]:
    """
    Поддерживает:
    - обычный JSON: [ {...}, {...}, ... ]
    - JSON Lines: по одному объекту в строке
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":  # обычный JSON-массив
        data = json.loads(text)
        assert isinstance(data, list), "Ожидается массив объектов"
        return data
    # jsonl
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_y(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    y_true = [r["advert_category"] for r in results if "advert_category" in r]
    y_pred = [r["predicted_category"] for r in results if "advert_category" in r]
    return y_true, y_pred

def _to_py_scalar(x):
    # безопасная сериализация numpy/NaN
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (float, int)):
        return x
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x

def _round_map(d: dict, ndigits: int = 6) -> dict:
    out = {}
    for k, v in d.items():
        v = _to_py_scalar(v)
        if isinstance(v, float) and math.isfinite(v):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out

def build_global_label_index(all_exp: dict[str, dict]) -> tuple[list[str], dict[str,int]]:
    # алфавитный список всех меток (истина ∪ предсказания по всем экспериментам)
    labels = sorted(set().union(*[set(res["per_class"]["label"]) for res in all_exp.values()]))
    l2i = {lab: i for i, lab in enumerate(labels)}
    return labels, l2i

def save_label_index(labels: list[str], out_dir: str | Path) -> Path:
    out = _ensure_dir(out_dir)
    df = pd.DataFrame({"idx": range(len(labels)), "label": labels})
    path = out / "labels_index.csv"
    df.to_csv(path, index=False)
    return path

def reindex_confusion(cm_local: np.ndarray, local_labels: list[str],
                      global_labels: list[str]) -> np.ndarray:
    idx_map = {lab: i for i, lab in enumerate(global_labels)}
    size = len(global_labels)
    cm_global = np.zeros((size, size), dtype=int)
    loc_idx = {lab: i for i, lab in enumerate(local_labels)}
    for lab_t, i_t in loc_idx.items():
        for lab_p, i_p in loc_idx.items():
            gi = idx_map[lab_t]
            gj = idx_map[lab_p]
            cm_global[gi, gj] = cm_local[i_t, i_p]
    return cm_global

def plot_confusion_heatmap_indices(cm: np.ndarray, title: str,
                                   tick_step: int = 5, normalize: str | None = "true"):
    arr = cm.astype(float)
    if normalize == "true":
        denom = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom != 0.0)
        vmin, vmax = 0.0, 1.0
        cbar_label = "share"
    elif normalize == "pred":
        denom = arr.sum(axis=0, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom != 0.0)
        vmin, vmax = 0.0, 1.0
        cbar_label = "share"
    else:
        vmin, vmax = 0, arr.max() if arr.size else 1
        cbar_label = "count"

    n = arr.shape[0]
    fig_w = min(14, max(6, n * 0.20))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, cmap="Blues", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Predicted (idx)")
    ax.set_ylabel("True (idx)")

    ax.set_xticks(range(0, n, tick_step))
    ax.set_yticks(range(0, n, tick_step))
    ax.set_xticklabels(range(0, n, tick_step))
    ax.set_yticklabels(range(0, n, tick_step))

    # сетка-решётка
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label)
    fig.tight_layout()
    return fig, ax

def save_confusion_images_indexed(exp_name: str, cm_local: np.ndarray, local_labels: list[str],
                                  global_labels: list[str], out_dir: str | Path,
                                  tick_step: int = 5) -> tuple[Path, Path]:
    out = _ensure_dir(Path(out_dir) / exp_name)
    cm_g = reindex_confusion(cm_local, local_labels, global_labels)

    fig, _ = plot_confusion_heatmap_indices(cm_g, title=f"{exp_name} — Confusion (abs, idx)",
                                            tick_step=tick_step, normalize=None)
    abs_path = out / f"{exp_name}_confusion_abs_idx.png"
    fig.savefig(abs_path, bbox_inches="tight", dpi=150); plt.close(fig)

    fig, _ = plot_confusion_heatmap_indices(cm_g, title=f"{exp_name} — Confusion (row-norm, idx)",
                                            tick_step=tick_step, normalize="true")
    norm_path = out / f"{exp_name}_confusion_row_norm_idx.png"
    fig.savefig(norm_path, bbox_inches="tight", dpi=150); plt.close(fig)

    return abs_path, norm_path

def analyze_experiment(path: str | Path) -> Dict[str, Any]:
    results = _load_results(path)
    y_true, y_pred = _extract_y(results)
    n_samples = len(y_true)

    # Все встречающиеся метки (истина ∪ предсказания) в фиксированном порядке
    labels = sorted(set(y_true) | set(y_pred))

    # Глобальные метрики
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # TP / FP / FN в агрегате (для многоклассовой задачи)
    # TP — количество точных совпадений; суммарные FP и FN равны числу ошибок
    tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    err = n_samples - tp_total
    fp_total = err
    fn_total = err

    summary = {
        "n_classes": len(labels),
        "n_samples": n_samples,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "precision": prec_micro,  # micro-precision == micro-recall == micro-F1 при мультиклассе с 1 меткой
        "recall": rec_micro,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        # Дополнительно можно вернуть и другие усреднения:
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "micro_f1": f1_micro,
    }

    # Поклассовые метрики
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Нормировка по строкам (доли внутри истинного класса)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm_true = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Для каждого класса считаем TP/FP/FN из матрицы
    tp_vec = np.diag(cm)
    fp_vec = cm.sum(axis=0) - tp_vec
    fn_vec = cm.sum(axis=1) - tp_vec

    per_class = pd.DataFrame({
        "label": labels,
        "support": support.astype(int),
        "precision": p_cls,
        "recall": r_cls,
        "f1": f1_cls,
        "tp": tp_vec.astype(int),
        "fp": fp_vec.astype(int),
        "fn": fn_vec.astype(int),
    }).sort_values(by=["f1", "support"], ascending=[True, False]).reset_index(drop=True)

    return {
        "summary": summary,
        "per_class": per_class,
        "labels": labels,
        "cm": cm,
        "cm_norm_true": cm_norm_true,
    }

def render_summary_table(summary: dict) -> pd.DataFrame:
    """Возвращает DataFrame с краткой сводкой по эксперименту."""
    order = [
        "n_classes", "n_samples",
        "macro_f1", "weighted_f1",
        "precision", "recall",
        "tp", "fp", "fn",
        # можно держать под рукой и эти поля
        "macro_precision", "macro_recall",
        "weighted_precision", "weighted_recall",
        "micro_f1",
    ]
    row = {k: summary.get(k) for k in order if k in summary}
    return pd.DataFrame([row])

def plot_confusion_heatmap(cm: np.ndarray, labels: list[str], title: str = "Confusion matrix (normalized)", normalize: str = "true"):
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
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=arr.max() if normalize is None else 1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90)
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

def plot_per_class_table(per_class: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """
    Возвращает аккуратную таблицу per-class метрик в алфавитном порядке.
    top_k: можно ограничить первыми K классами (после сортировки по label).
    """
    cols = ["label", "support", "precision", "recall", "f1", "tp", "fp", "fn"]
    df = per_class[cols].sort_values("label").reset_index(drop=True)
    if top_k:
        df = df.head(top_k)
    return df

def summarize_experiments(exp_results: Dict[str, dict]) -> pd.DataFrame:
    """
    Возвращает сводную таблицу глобальных метрик по экспериментам.
    """
    rows = []
    for name, res in exp_results.items():
        s = res["summary"]
        rows.append({
            "experiment": name,
            "n_classes": s["n_classes"],
            "n_samples": s["n_samples"],
            "macro_f1": s["macro_f1"],
            "weighted_f1": s["weighted_f1"],
            "precision": s["precision"],  # micro
            "recall": s["recall"],        # micro
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
        })
    df = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
    return df

def per_class_metrics_by_experiment(exp_results: Dict[str, dict]) -> pd.DataFrame:
    """
    Объединяет per-class метрики всех экспериментов в одну широкую таблицу.
    """
    # универсальное множество меток
    labels = sorted(set().union(*[set(res["per_class"]["label"]) for res in exp_results.values()]))
    dfs = []
    for name, res in exp_results.items():
        df = res["per_class"].set_index("label").reindex(labels)
        # добавим префикс колонок
        df = df.add_prefix(f"{name}_")
        # вернём столбец label для merge
        df = df.reset_index().rename(columns={"index": "label"})
        dfs.append(df)
    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="label", how="outer")
    return out.sort_values("label").reset_index(drop=True)

def deltas_table(exp_results: Dict[str, dict], base: str, other: str) -> pd.DataFrame:
    """
    Возвращает таблицу дельт per-class между двумя экспериментами: other − base.
    """
    wide = per_class_metrics_by_experiment(exp_results)
    # возьмём F1/Precision/Recall + support базового и целевого
    cols = ["label",
            f"{base}_f1", f"{other}_f1",
            f"{base}_precision", f"{other}_precision",
            f"{base}_recall", f"{other}_recall",
            f"{base}_support", f"{other}_support"]
    df = wide[cols].copy()
    df["dF1"] = df[f"{other}_f1"] - df[f"{base}_f1"]
    df["dPrec"] = df[f"{other}_precision"] - df[f"{base}_precision"]
    df["dRec"] = df[f"{other}_recall"] - df[f"{base}_recall"]
    return df.sort_values("label").reset_index(drop=True)

def plot_macro_weighted_over_time(summary_df: pd.DataFrame, title: str = "Macro/Weighted F1 over experiments"):
    """
    Линейный график динамики Macro/Weighted F1 по экспериментам.
    """
    x = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, summary_df["macro_f1"], marker="o", label="Macro F1")
    ax.plot(x, summary_df["weighted_f1"], marker="o", label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["experiment"])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_delta_f1_bars(delta_df: pd.DataFrame, title: str):
    """
    Барчарт ΔF1 по категориям (отсортирован по величине дельты).
    """
    dd = delta_df[["label", "dF1"]].dropna().sort_values("dF1", ascending=True)
    n = len(dd)
    fig_h = min(12, max(5, 0.25 * n))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(dd["label"], dd["dF1"])
    ax.set_xlabel("ΔF1")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=1)
    fig.tight_layout()
    return fig, ax

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_summary_csv(exp_name: str, summary: dict, out_dir: str | Path) -> Path:
    out = _ensure_dir(Path(out_dir) / exp_name)
    df = render_summary_table(summary)
    path = out / f"{exp_name}_summary.csv"
    df.to_csv(path, index=False)
    return path

def save_per_class_csv(exp_name: str, per_class: pd.DataFrame, out_dir: str | Path) -> Path:
    out = _ensure_dir(Path(out_dir) / exp_name)
    df = per_class.sort_values("label").reset_index(drop=True)
    path = out / f"{exp_name}_per_class.csv"
    df.to_csv(path, index=False)
    return path

def save_confusion_images(exp_name: str, cm: np.ndarray, labels: list[str], out_dir: str | Path) -> tuple[Path, Path]:
    out = _ensure_dir(Path(out_dir) / exp_name)
    # Абсолюты
    fig, ax = plot_confusion_heatmap(cm, labels, title=f"{exp_name} — Confusion (abs)", normalize=None)
    abs_path = out / f"{exp_name}_confusion_abs.png"
    fig.savefig(abs_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Нормировка по истинному классу
    fig, ax = plot_confusion_heatmap(cm, labels, title=f"{exp_name} — Confusion (row-normalized)", normalize="true")
    norm_path = out / f"{exp_name}_confusion_row_norm.png"
    fig.savefig(norm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return abs_path, norm_path

def save_global_summary(all_exp: dict[str, dict], out_dir: str | Path) -> Path:
    out = _ensure_dir(out_dir)
    df = summarize_experiments(all_exp)
    path = out / "global_summary.csv"
    df.to_csv(path, index=False)
    return path

def save_macro_weighted_plot(all_exp: dict[str, dict], out_dir: str | Path) -> Path:
    out = _ensure_dir(out_dir)
    summary_df = summarize_experiments(all_exp)
    fig, ax = plot_macro_weighted_over_time(summary_df)
    path = out / "macro_weighted_over_time.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def save_delta_tables_and_plots(all_exp: dict[str, dict], base: str, other: str, out_dir: str | Path) -> tuple[Path, Path]:
    out = _ensure_dir(out_dir)
    ddf = deltas_table(all_exp, base=base, other=other)
    csv_path = out / f"deltas_{other}-vs-{base}.csv"
    ddf.to_csv(csv_path, index=False)

    fig, ax = plot_delta_f1_bars(ddf, title=f"ΔF1 ({other} − {base}) by class")
    png_path = out / f"deltas_{other}-vs-{base}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return csv_path, png_path

def save_experiment_json(exp_name: str, analysis: dict, out_dir: str | Path,
                         ndigits: int = 6) -> Path:
    """
    Сохраняет JSON с полями:
      {
        "summary": {...},
        "per_class": [
          {"label": ..., "n_samples": ..., "f1": ..., "precision": ..., "recall": ..., "tp": ..., "fp": ..., "fn": ...},
          ...
        ]
      }
    """
    out = _ensure_dir(Path(out_dir) / exp_name)

    # summary
    summary = _round_map(analysis["summary"], ndigits)

    # per-class → требуемые поля и переименование support → n_samples
    per_class: pd.DataFrame = analysis["per_class"].copy()
    per_class = per_class.sort_values("label")
    per_class = per_class.rename(columns={"support": "n_samples"})
    cols = ["label", "n_samples", "f1", "precision", "recall", "tp", "fp", "fn"]

    per_class_records = []
    for _, row in per_class[cols].iterrows():
        rec = {c: _to_py_scalar(row[c]) for c in cols}
        per_class_records.append(_round_map(rec, ndigits))

    payload = {"summary": summary, "per_class": per_class_records}

    path = out / f"{exp_name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def make_markdown_report(all_exp: dict[str, dict], out_dir: str | Path, base="Exp1", middle="Exp2", last="Exp3") -> Path:
    out = _ensure_dir(out_dir)
    lines = []
    lines.append(f"# Отчёт по классификации\n")

    # Глобальные метрики
    glob_csv = save_global_summary(all_exp, out)
    macro_png = save_macro_weighted_plot(all_exp, out)
    lines.append("## Глобальные метрики")
    lines.append(f"- [global_summary.csv]({glob_csv.name})")
    lines.append(f"![Macro/Weighted F1]({macro_png.name})\n")

    # По экспериментам
    for name, res in all_exp.items():
        exp_dir = Path(out) / name
        _ensure_dir(exp_dir)
        sum_csv = save_summary_csv(name, res["summary"], out)
        pc_csv = save_per_class_csv(name, res["per_class"], out)
        abs_png, norm_png = save_confusion_images(name, res["cm"], res["labels"], out)
        json_path = save_experiment_json(name, res, out)  # ← новый JSON


        lines.append(f"## {name}")
        lines.append(f"- [summary]({(Path(name)/Path(sum_csv).name).as_posix()})")
        lines.append(f"- [per-class]({(Path(name)/Path(pc_csv).name).as_posix()})")
        lines.append(f"- [full.json]({(Path(name)/Path(json_path).name).as_posix()})")
        lines.append(f"![Confusion abs]({(Path(name)/Path(abs_png).name).as_posix()})")
        lines.append(f"![Confusion row-norm]({(Path(name)/Path(norm_png).name).as_posix()})\n")

    # Дельты
    d21_csv, d21_png = save_delta_tables_and_plots(all_exp, base=base, other=middle, out_dir=out)
    d31_csv, d31_png = save_delta_tables_and_plots(all_exp, base=base, other=last, out_dir=out)
    lines.append("## Дельты между экспериментами")
    lines.append(f"- [Δ {middle}−{base}]({Path(d21_csv).name})")
    lines.append(f"![ΔF1 {middle}−{base}]({Path(d21_png).name})")
    lines.append(f"- [Δ {last}−{base}]({Path(d31_csv).name})")
    lines.append(f"![ΔF1 {last}−{base}]({Path(d31_png).name})\n")

    report_path = Path(out) / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

def run_full_analysis(exp_files: dict[str, str | Path], out_dir: str | Path = "analysis_out") -> Path:
    """
    exp_files: {"Exp1": "exp1.json", "Exp2": "exp2.json", "Exp3": "exp3.json"}
    Возвращает путь к Markdown-отчёту.
    """
    from pprint import pprint
    from collections import OrderedDict

    # Фиксируем порядок экспериментов так, как задан в словаре
    exp_ordered = OrderedDict(exp_files)

    # 1) Анализ каждого эксперимента
    analyzed: dict[str, dict] = {}
    for name, path in exp_ordered.items():
        res = analyze_experiment(path)
        analyzed[name] = res

    # 2) Сводный Markdown-отчёт
    report = make_markdown_report(analyzed, out_dir=out_dir,
                                  base=list(exp_ordered.keys())[0],
                                  middle=list(exp_ordered.keys())[1],
                                  last=list(exp_ordered.keys())[-1])
    return report


class ReportService:
    def run(self, exp_files: dict[str, Any], out_dir: str) -> str | Path:
        return run_full_analysis(exp_files, out_dir)

