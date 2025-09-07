from core.domain.advert import Advert
from core.domain.category import Category


def category_to_prompt_ctx(cat: Category) -> dict[str, object]:
    return {
        "id": cat.id,
        "title": cat.title,
        "bow_str": cat.bow_str,
        "tf_idf_str": cat.tf_idf_str,
    }

def advert_to_prompt_ctx(adv: Advert) -> dict[str, object]:
    return {
        "id": adv.advert_id or 0,
        "title": adv.advert_title,
        "text": adv.advert_text,
        "summary": adv.advert_summary or "",
        "category_id": adv.category_id,
        "category_title": adv.category_title,
    }
