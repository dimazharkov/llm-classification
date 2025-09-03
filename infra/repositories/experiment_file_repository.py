from dataclasses import is_dataclass, asdict

from pydantic import BaseModel

from core.contracts.experiment_repository import ExperimentRepository
from infra.storage.os_helper import save_to_disc


class ExperimentFileRepository(ExperimentRepository):
    def __init__(self, path: str) -> None:
        self.path = path

    def save_list(self, data_list: list[dict]) -> None:
        prepared = []
        for item in data_list:
            if isinstance(item, BaseModel):
                prepared.append(item.model_dump(mode="json"))
            elif is_dataclass(item):
                prepared.append(asdict(item))
            else:
                prepared.append(item)
        save_to_disc(prepared, self.path)