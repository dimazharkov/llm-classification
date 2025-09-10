from typing import Any

from src.infra.repositories.base_file_repository import BaseFileRepository


class JsonFileRepository(BaseFileRepository[dict[str, Any]]):
    def _mapper(self, item: Any) -> dict[str, Any]:
        return item.as_dict()
