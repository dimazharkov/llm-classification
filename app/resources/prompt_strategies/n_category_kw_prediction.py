from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class NCategoryKwPrediction(PromptStrategy):
    name = "n_category_kw_prediction"
    
    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an advertisement classifier. "
            "Analyze the advertisement text and select the three most suitable categories from the list.\n\n"
            "Advertisement:\n"
            f"{ctx['advert_title']!r} {ctx['advert_text']!r}\n\n"
            "List of categories with keywords:\n"
            f"{ctx['categories_with_keywords']!r}\n\n"
            "Each category is provided in the format:\n"
            "- category: keyword1, keyword2, ...\n\n"
            "Answer strictly with a list of three categories, one per line, without numbering, quotes, dashes, keywords, formatting, or explanations.\n\n"
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        ...