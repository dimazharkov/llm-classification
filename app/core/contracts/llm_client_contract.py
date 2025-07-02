
from typing import Protocol


class LLMClientContract(Protocol):
    def generate(self, prompt: str, instructions: str = None) -> str: ...
