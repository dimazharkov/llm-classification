from typing import Optional, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, instructions: Optional[str] = None) -> str: ...
