from typing import Optional

from pydantic import BaseModel


class Advert(BaseModel):
    category_id: int
    category_title: str
    advert_title: str
    advert_text: str
    advert_summary: Optional[str] = None

