from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class CategoryPairPrediction(PromptStrategy[dict[str, Any], str]):
    name = "category_pair_prediction"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an expert in advertisement classification. "
            "You are given an advertisement, two categories, and a description of their differences.\n\n"
            "Advertisement:\n"
            f"{ctx['advert']['summary']!r}\n\n"
            "Categories:\n"
            f"- {ctx['category1']['title']}: {ctx['category1']['tf_idf_str']}\n"
            f"- {ctx['category2']['title']}: {ctx['category2']['tf_idf_str']}\n\n"
            f"Description of differences:\n{ctx['difference']}\n\n"
            "Rules:\n"
            f"- Choose exactly one label: «{ctx['category1']['title']}», «{ctx['category2']['title']}», or «other».\n"
            "- Match the meaning of the advertisement strictly with the provided features.\n"
            "- Ignore brands, condition, price/ціна, delivery, phone numbers, toponyms, HTML, and irrelevant words.\n"
            "- If the type of object/product/service, its purpose, or its context does not fit either category — answer «other».\n"
            "- If the match is weak, contradictory, or only general words overlap — answer «other».\n"
            "- Internally assess the match with each category (0–1); if both scores are low (<0.6) — answer «other». Do not output the scores.\n"
            "- Always use the exact category names; never write “first/second category”.\n\n"
            f"Answer strictly with a single lowercase word without quotes: «{ctx['category1']['title']}» or «{ctx['category2']['title']}» or «other»."
        )

    def parse_response(self, raw: str) -> str:
        return raw.strip()
