from typing import Any, Protocol


class ExperimentRepository(Protocol):
    def save_list(self, data_list: list[Any]) -> None: ...
