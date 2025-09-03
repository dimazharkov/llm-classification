from typing import Protocol

from core.domain.advert import Advert


class AdvertRepository(Protocol):
    def get_all(self) -> list[Advert]: ...
    def get_all_filtered(self) -> list[Advert]: ...