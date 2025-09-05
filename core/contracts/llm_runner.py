from typing import Any, Protocol


class LLMRunner(Protocol):
    def run(self, prompt_name: str, ctx: Any) -> Any: ...
