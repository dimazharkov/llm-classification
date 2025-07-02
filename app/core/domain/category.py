from typing import Optional

from pydantic import BaseModel


class Category(BaseModel):
    id: int
    title: str
    bow: Optional[list[str]] = None
    tf_idf: Optional[list[str]] = None
