from pydantic import BaseModel
from app.core.domain.category import Category


class CategoryDiff(BaseModel):
    category1: Category
    category2: Category
    difference: str