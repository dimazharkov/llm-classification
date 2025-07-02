import random
from collections import defaultdict
from typing import Iterable

from app.core.domain.advert import Advert
from app.core.domain.advert_raw import AdvertRaw


class PreprocessAdvertsUseCase:
    def __init__(self, raw_adverts: list[AdvertRaw]) -> None:
        self._raw_adverts = raw_adverts

    def run(self, categories_ids: Iterable[int] = None, max_per_category: int = 20) -> list[Advert]:
        if categories_ids is not None:
            categories_ids = set(categories_ids)

        random.shuffle(self._raw_adverts)

        result = []
        category_counter = defaultdict(int)

        for raw in self._raw_adverts:
            if categories_ids is not None and raw.category_id not in categories_ids:
                continue

            if category_counter[raw.category_id] >= max_per_category:
                continue

            advert = Advert(
                category_id=raw.category_id,
                category_title=raw.category_title,
                advert_title=raw.title,
                advert_text=raw.text
            )
            result.append(advert)
            category_counter[raw.category_id] += 1

        return result