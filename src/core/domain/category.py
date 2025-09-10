from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Category:
    id: int
    title: str
    bow: list[str] | None = None
    tf_idf: list[str] | None = None

    @property
    def bow_str(self) -> str:
        return ", ".join(self.bow or [])

    @property
    def tf_idf_str(self) -> str:
        return ", ".join(self.tf_idf or [])
