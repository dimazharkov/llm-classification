import json
from pathlib import Path
from typing import Any, Union

import pandas as pd
from matplotlib.figure import Figure

from src.app.config.config import config

PathLike = Union[str, Path]

def resolve_storage_path(file_path: PathLike) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    relative_path = Path(str(file_path).lstrip("/\\"))
    return config.static_path / relative_path


def save_to_disc(data: pd.DataFrame | dict[str, Any], file_path: str | Path, indent: int = 4) -> None:
    full_file_path = resolve_storage_path(file_path)
    full_file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_json(full_file_path, orient="records", indent=indent, force_ascii=False)
    else:
        with open(full_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)


def save_plot_to_disc(plt: Figure, file_path: str | Path, dpi: int = 300) -> None:
    full_file_path = resolve_storage_path(file_path)
    full_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_file_path, dpi=dpi)


def load_from_disc(file_path: str | Path) -> Any:
    full_file_path = resolve_storage_path(file_path)

    if not full_file_path.exists():
        raise FileNotFoundError(f"Missing file: {full_file_path}")

    with open(full_file_path, encoding="utf-8") as f:
        return json.load(f)


def load_df_from_disc(file_path: str | Path) -> pd.DataFrame:
    data = load_from_disc(file_path)
    return pd.DataFrame(data)
