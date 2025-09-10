from pydantic import BaseModel


class AdvertSchema(BaseModel):
    advert_id: int
    category_id: int
    category_title: str
    advert_title: str
    advert_text: str
    advert_summary: str
