from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

RepositoryItemType = TypeVar("RepositoryItemType")


@runtime_checkable
class FileRepository(Protocol[RepositoryItemType]):
    def read(self) -> list[RepositoryItemType]: ...
    def save_list(self, data_list: Iterable[Any]) -> None: ...
