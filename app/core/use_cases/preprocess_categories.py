from collections.abc import Iterable

from app.core.contracts.use_case_contract import UseCaseContract
from app.core.domain.category import Category
from app.core.dto.category_raw import CategoryRaw


class PreprocessCategoriesUseCase(UseCaseContract):
    def __init__(self, advert_categories: dict[int, CategoryRaw]) -> None:
        self._advert_categories = advert_categories

    def run(self, selected_ids: Iterable[int] = None) -> list[Category]:
        selected = (
            list(self._advert_categories.items())
            if selected_ids is None
            else [
                (category_id, category)
                for category_id, category in self._advert_categories.items()
                if category_id in selected_ids
            ]
        )

        return [Category(id=category_id, title=category.category_title) for category_id, category in selected]
