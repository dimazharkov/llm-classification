from typing import Any

from src.core.contracts.file_repository import FileRepository


class ExperimentRepository(FileRepository[dict[str, Any]]): ...
