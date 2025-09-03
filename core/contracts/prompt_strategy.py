from typing import Protocol, Any, TypeVar, Generic

StrategyContextType = TypeVar("StrategyContextType", contravariant=True)
StrategyResultType = TypeVar("StrategyResultType", covariant=True)


class PromptStrategy(Protocol, Generic[StrategyContextType, StrategyResultType]):
    name: str
    def build_prompt(self, ctx: StrategyContextType) -> str: ...
    def parse_response(self, raw: str) -> StrategyResultType: ...