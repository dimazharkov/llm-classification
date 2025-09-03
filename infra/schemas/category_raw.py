from pydantic import BaseModel


class CategoryRaw(BaseModel):
    category_title: str
    parent_id: int
    items: int
