from collections.abc import Iterable

from app.core.dto.category_diff import CategoryDiff
from app.shared.helpers.category_pair_utils import normalize_pair
from app.shared.types.category_id_pair import CategoryIdPair


class CategoryPairDiffRepository:
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[CategoryIdPair, CategoryDiff] = {}

    def all(self) -> Iterable[tuple[CategoryIdPair, CategoryDiff]]:
        return self._data.items()

    def all_titles(self) -> Iterable[str]:
        titles = set()
        for _, diff in self.all():
            titles.add(diff.category1.title)
            titles.add(diff.category2.title)
        return sorted(titles)

    def get(self, pair: CategoryIdPair) -> CategoryDiff | None:
        return self._data.get(normalize_pair(pair))

    def add(self, pair: CategoryIdPair, diff_text: CategoryDiff) -> None:
        self._data[normalize_pair(pair)] = diff_text

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        count = len(self._data)
        preview = "\n".join(
            f"{k}: {v.difference[:60]}..." for k, v in list(self._data.items())[:3]
        )
        return f"<{class_name} with {count} pairs>\nPreview:\n{preview}"
