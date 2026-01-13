from typing import Any, Optional

from src.core.contracts.category_repository import CategoryRepository
from src.core.domain.category import Category
from src.infra.mappers.category_mapper import schema_to_category
from src.infra.schemas.category_schema import CategorySchema
from src.infra.storage.os_helper import load_from_disc


class CategoryFileRepository(CategoryRepository):
    def __init__(self, path: str) -> None:
        self.data: list[Category] = self._load(path)

    def get_all(self) -> list[Category]:
        return self.data

    def get_all_with_kw(self, method: Optional[str] = None) -> str:
        # return "\n".join(f"- {c.title}: {', '.join(c.bow or [])}" for c in self.get_all())
        if method == "tfidf":
            return self.get_all_with_tfidf()
        return self.get_all_with_bw()

    def get_all_with_bw(self) -> str:
        return "\n".join(f"- {c.title}: {', '.join(c.bow or [])}" for c in self.get_all())

    def get_all_with_tfidf(self) -> str:
        return "\n".join(f"- {c.title}: {', '.join(c.tf_idf or [])}" for c in self.get_all())

    def get_titles_str(self) -> str:
        return ", ".join(category.title for category in self.data)

    def get_by_titles(self, titles_list: list[str]) -> list[Category]:
        return [category for category in self.get_all() if category.title in titles_list]

    def _load(self, path: str) -> list[Category]:
        raw: list[dict[str, Any]] = load_from_disc(path)
        return [schema_to_category(CategorySchema.model_validate(c)) for c in raw]
