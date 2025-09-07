from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class CategoryPrediction(PromptStrategy[dict[str, Any], str]):
    name = "category_prediction"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an advertisement classifier. "
            "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
            "Advertisement:\n"
            f"{ctx['advert']['text']!r}\n\n"
            f"Categories:\n{ctx['category_titles']}\n\n"
            "Answer strictly with the category name only."
        )

    def parse_response(self, raw: str) -> str:
        return raw.strip()
