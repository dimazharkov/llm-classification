from pydantic import BaseModel


class Category(BaseModel):
    id: int
    title: str
    bow: list[str] | None = None
    tf_idf: list[str] | None = None
