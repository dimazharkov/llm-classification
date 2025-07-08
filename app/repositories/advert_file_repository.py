from collections import defaultdict

from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.helpers.os_helper import load_from_disc


class AdvertFileRepository:
    def __init__(self, path: str):
        self.data: list[Advert] = self._load(path)
        self.category_index: dict[int, list[Advert]] = self._build_category_index(self.data)

    def get(self) -> list[Advert]:
        return self.data

    def get_adverts_by_category(self, category: Category) -> list[Advert]:
        return self.category_index.get(category.id, [])

    def _load(self, path: str) -> list[Advert]:
        raw = load_from_disc(path)
        return [Advert.model_validate(c) for c in raw]

    def _build_category_index(self, adverts: list[Advert]) -> dict[int, list[Advert]]:
        index = defaultdict(list)
        for advert in adverts:
            index[advert.category_id].append(advert)
        return dict(index)