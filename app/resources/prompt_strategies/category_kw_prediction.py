from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class CategoryKwPrediction(PromptStrategy[dict[str, Any], str]):
    name = "category_kw_prediction"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an advertisement classifier. "
            "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
            "Advertisement:\n"
            f"{ctx['advert_text']!r}\n\n"
            "Categories with keywords:\n"
            f"{ctx['categories_with_kw']}\n\n"
            "Each category is provided in the format:\n"
            "- category: keyword1, keyword2, ...\n\n"
            "Answer strictly with the category name only."
        )

    def parse_response(self, raw: str) -> str:
        return raw.strip()