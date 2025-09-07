from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Category:
    id: int
    title: str
    bow: Optional[list[str]] = None
    tf_idf: Optional[list[str]] = None

    @property
    def bow_str(self) -> str:
        return ", ".join(self.bow or [])

    @property
    def tf_idf_str(self) -> str:
        return ", ".join(self.tf_idf or [])