from typing import Protocol, TypeVar, runtime_checkable

StrategyContextType = TypeVar("StrategyContextType", contravariant=True)
StrategyResultType = TypeVar("StrategyResultType", covariant=True)


# class PromptStrategy(Protocol, Generic[StrategyContextType, StrategyResultType]):
@runtime_checkable
class PromptStrategy(Protocol[StrategyContextType, StrategyResultType]):
    name: str

    def build_prompt(self, ctx: StrategyContextType) -> str: ...
    def parse_response(self, raw: str) -> StrategyResultType: ...
