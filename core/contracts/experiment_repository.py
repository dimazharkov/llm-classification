from typing import Protocol, Any


class ExperimentRepository(Protocol):
    def save_list(self, data_list: list[Any]) -> None: ...