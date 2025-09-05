from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Advert:
    category_id: int
    category_title: str
    advert_title: str
    advert_text: str
    advert_summary: str | None = None
    advert_id: int | None = None
