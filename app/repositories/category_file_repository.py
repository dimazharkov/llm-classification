from typing import Optional

from app.core.domain.category import Category
from app.helpers.os_helper import load_from_disc


class CategoryFileRepository():
    def __init__(self, path: str):
        self.data: list[Category] = self._load(path)
        self.title_index: dict[str, Category] = {cat.title: cat for cat in self.data}


    def get(self) -> list[Category]:
        return self.data

    def get_category_by_title(self, title: str) -> Optional[Category]:
        return self.title_index.get(title)

    def _load(self, path: str) -> list[Category]:
        raw = load_from_disc(path)
        return [Category.model_validate(c) for c in raw]
