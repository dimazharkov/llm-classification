from typing import Any

from pydantic import BaseModel

from app.helpers.os_helper import save_to_disc


class JsonSaver:
    def __init__(self, path: str) -> None:
        self.path = path

    def save_list(self, payload: list[Any]) -> None:
        prepared = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in payload]
        save_to_disc(prepared, self.path)
