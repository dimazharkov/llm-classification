from typing import Any

from core.contracts.advert_repository import AdvertRepository
from core.domain.advert import Advert
from infra.mappers.advert_mapper import schema_to_advert
from infra.schemas.advert_schema import AdvertSchema
from infra.storage.os_helper import load_from_disc


class AdvertFileRepository(AdvertRepository):
    def __init__(self, path: str) -> None:
        self.data: list[Advert] = self._load(path)

    def get_all(self) -> list[Advert]:
        return self.data

    def get_all_filtered(self) -> list[Advert]:
        excluded = [
            111, 162, 154, 112, 236, 110, 218, 319, 189, 313, 248, 173, 34, 114, 54, 230, 97, 290, 67, 38, 62, 77,
            87, 11, 165, 72, 27, 225, 143, 350, 133, 5, 3, 269, 264, 252, 289, 243, 229, 200, 151, 53
        ]

        return [advert for advert in self.data if advert.advert_id not in excluded]

    def _load(self, path: str) -> list[Advert]:
        raw: list[dict[str, Any]] = load_from_disc(path)
        return [schema_to_advert(AdvertSchema.model_validate(c)) for c in raw]
