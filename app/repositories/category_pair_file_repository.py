from __future__ import annotations

import json
import atexit
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

from app.helpers.os_helper import save_to_disc, load_from_disc
from app.shared.helpers.category_pair_utils import split_pair_key, normalize_pair
from app.shared.types.category_id_pair import CategoryIdPair


class CategoryPairFileRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._data: Dict[CategoryIdPair, str] = {}

        atexit.register(self.save)

    def all(self) -> Iterable[Tuple[CategoryIdPair, str]]:
        return self._data.items()

    def get(self, pair: CategoryIdPair) -> Optional[str]:
        return self._data.get(normalize_pair(pair))

    def add(self, pair: CategoryIdPair, diff_text: str) -> None:
        self._data[normalize_pair(pair)] = diff_text

    def save(self) -> None:
        serializable = {f"{a}:{b}": txt for (a, b), txt in self._data.items()}
        save_to_disc(serializable, self._path)

    def load(self) -> None:
        try:
            raw = load_from_disc(self._path)
        except FileNotFoundError:
            self._data = {}
            return
        self._data = {split_pair_key(k): v for k, v in raw.items()}
