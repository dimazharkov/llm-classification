from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class CategoryPairPrediction(PromptStrategy):
    name = "category_pair_prediction"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an expert in advertisement classification. "
            "You are given an advertisement, two categories, and a description of their differences.\n\n"
            "Advertisement:\n"
            f"{ctx['advert']['advert_summary']!r}\n\n"
            "Categories:\n"
            "- {category1.title}: {category1_keywords}\n"
            "- {category2.title}: {category2_keywords}\n\n"
            "Description of differences:\n{difference}\n\n"
            "Rules:\n"
            "- Choose exactly one label: «{category1.title}», «{category2.title}», or «other».\n"
            "- Match the meaning of the advertisement strictly with the provided features.\n"
            "- Ignore brands, condition, price/ціна, delivery, phone numbers, toponyms, HTML, and irrelevant words.\n"
            "- If the type of object/product/service, its purpose, or its context does not fit either category — answer «other».\n"
            "- If the match is weak, contradictory, or only general words overlap — answer «other».\n"
            "- Internally assess the match with each category (0–1); if both scores are low (<0.6) — answer «other». Do not output the scores.\n"
            "- Always use the exact category names; never write “first/second category”.\n\n"
            "Answer strictly with a single lowercase word without quotes: «{category1.title}» or «{category2.title}» or «other»."
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        ...