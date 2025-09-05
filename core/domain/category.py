from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Category:
    id: int
    title: str
    bow: list[str] | None = None
    tf_idf: list[str] | None = None
