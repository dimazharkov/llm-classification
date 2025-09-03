from typing import Dict, Any

from core.contracts.prompt_strategy import PromptStrategy


class PromptRegistry:
    def __init__(self, strategies: list[PromptStrategy[Any, Any]]):
        self.prompts: Dict[str, PromptStrategy[Any, Any]] = {s.name: s for s in strategies}

    def get(self, name: str) -> PromptStrategy[Any, Any]:
        return self.prompts[name]
