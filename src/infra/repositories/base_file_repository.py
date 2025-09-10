from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from src.core.contracts.file_repository import FileRepository, RepositoryItemType
from src.infra.storage.os_helper import load_from_disc, save_to_disc


def to_jsonable(item: Any) -> Any:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    return item


class BaseFileRepository(FileRepository[RepositoryItemType]):
    def __init__(self, path: str) -> None:
        self.path = path

    def save_list(self, data_list: Iterable[Any]) -> None:
        prepared = [to_jsonable(item) for item in data_list]
        save_to_disc(prepared, self.path)

    def read(self) -> list[RepositoryItemType]:
        raw_items = load_from_disc(self.path)
        return [self._mapper(item) for item in raw_items]

    def _mapper(self, item: Any) -> RepositoryItemType:
        raise NotImplementedError
