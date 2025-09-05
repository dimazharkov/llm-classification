from dataclasses import dataclass

from core.domain.category import Category


@dataclass(frozen=True, slots=True)
class CategoryDiff:
    category1: Category
    category2: Category
    difference: str
