from typing import Dict, Iterable, Tuple, Optional

from app.core.dto.category_diff import CategoryDiff
from app.shared.helpers.category_pair_utils import normalize_pair
from app.shared.types.category_id_pair import CategoryIdPair


class CategoryPairDiffRepository:
    __slots__ = ('_data',)

    def __init__(self) -> None:
        self._data: Dict[CategoryIdPair, CategoryDiff] = {}

    def all(self) -> Iterable[Tuple[CategoryIdPair, CategoryDiff]]:
        return self._data.items()

    def get(self, pair: CategoryIdPair) -> Optional[CategoryDiff]:
        return self._data.get(normalize_pair(pair))

    def add(self, pair: CategoryIdPair, diff_text: CategoryDiff) -> None:
        self._data[normalize_pair(pair)] = diff_text
