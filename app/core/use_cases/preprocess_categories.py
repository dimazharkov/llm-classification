from typing import Iterable, Dict

from app.core.domain.category import Category
from app.core.domain.category_raw import CategoryRaw


class PreprocessCategoriesUseCase:
    def __init__(self, advert_categories: Dict[int, CategoryRaw]) -> None:
        self._advert_categories = advert_categories

    def run(self, selected_ids: Iterable[int] = None) -> list[Category]:
        if selected_ids is None:
            selected = self._advert_categories.items()
        else:
            selected = [
                (category_id, category)
                for category_id, category in self._advert_categories.items()
                if category_id in selected_ids
            ]

        return [
            Category(id=category_id, title=category.category_title)
            for category_id, category in selected
        ]