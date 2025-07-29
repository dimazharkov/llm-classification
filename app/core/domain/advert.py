from typing import Optional

from pydantic import BaseModel


class Advert(BaseModel):
    advert_id: Optional[int] = None
    category_id: int
    category_title: str
    advert_title: str
    advert_text: str
    advert_summary: Optional[str] = None
