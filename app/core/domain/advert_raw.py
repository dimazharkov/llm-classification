from pydantic import BaseModel


class AdvertRaw(BaseModel):
    parent_title: str
    parent_id: int
    category_title: str
    category_id: int
    title: str
    text: str
