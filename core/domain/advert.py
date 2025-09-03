from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class Advert:
    category_id: int
    category_title: str
    advert_title: str
    advert_text: str
    advert_summary: Optional[str] = None
    advert_id: Optional[int] = None
