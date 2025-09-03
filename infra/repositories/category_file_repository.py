from typing import Any

from core.contracts.category_repository import CategoryRepository
from core.domain.category import Category
from infra.mappers.category_mapper import schema_to_category
from infra.schemas.category_schema import CategorySchema
from infra.storage.os_helper import load_from_disc


class CategoryFileRepository(CategoryRepository):
    def __init__(self, path: str) -> None:
        self.data: list[Category] = self._load(path)

    def get_all(self) -> list[Category]:
        return self.data

    def get_all_with_kw(self) -> str:
        return "\n".join(f"- {c.title}: {', '.join(c.bow or [])}" for c in self.get_all())

    def get_titles_str(self):
        return ", ".join(category.title for category in self.data)

    def _load(self, path: str) -> list[Category]:
        raw: list[dict[str, Any]] = load_from_disc(path)
        return [schema_to_category(CategorySchema.model_validate(c)) for c in raw]
