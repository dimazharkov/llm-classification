from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, instructions: str = None) -> str: ...
