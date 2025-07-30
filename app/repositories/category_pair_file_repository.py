from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

from app.helpers.os_helper import load_from_disc, save_to_disc
from app.shared.helpers.category_pair_utils import normalize_pair, split_pair_key

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from app.shared.types.category_id_pair import CategoryIdPair


class CategoryPairFileRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._data: dict[CategoryIdPair, str] = self.load()
        self.updated = False
        atexit.register(self.save)

    def all(self) -> Iterable[tuple[CategoryIdPair, str]]:
        return self._data.items()

    def get(self, pair: CategoryIdPair) -> str | None:
        return self._data.get(normalize_pair(pair))

    def add(self, pair: CategoryIdPair, diff_text: str) -> None:
        self._data[normalize_pair(pair)] = diff_text
        self.updated = True

    def save(self) -> None:
        if self.updated:
            serializable = {f"{a}:{b}": txt for (a, b), txt in self._data.items()}
            save_to_disc(serializable, self._path)
            self.updated = False

    def load(self) -> dict:
        try:
            raw = load_from_disc(self._path)
            return {split_pair_key(k): v for k, v in raw.items()}
        except FileNotFoundError:
            return {}
