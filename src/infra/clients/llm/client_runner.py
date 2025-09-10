from typing import Any

from src.core.contracts.llm_client import LLMClient
from src.core.contracts.prompt_strategy import PromptStrategy, StrategyContextType, StrategyResultType
from src.infra.clients.llm.prompt_registry import PromptRegistry


class LLMClientRunner(LLMClient):
    def __init__(self, llm_client: LLMClient, strat_registry: PromptRegistry):
        self.llm_client = llm_client
        self.strat_registry = strat_registry

    def run_with(
        self,
        strategy: PromptStrategy[StrategyContextType, StrategyResultType],
        ctx: StrategyContextType,
    ) -> StrategyResultType:
        prompt = strategy.build_prompt(ctx)
        response = self.llm_client.generate(prompt)
        return strategy.parse_response(response)

    def run(self, task: str, ctx: Any) -> Any:
        strategy = self.strat_registry.get(task)
        return self.run_with(strategy, ctx)
