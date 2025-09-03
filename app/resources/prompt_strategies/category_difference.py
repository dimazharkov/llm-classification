from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class CategoryDifference(PromptStrategy):
    name = "category_difference"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            "You are an expert in classifying and comparing advertisement categories. "
            "Based on the provided data, clearly define the main difference between the two categories.\n\n"
            f"Category **{ctx['category1']['title']}**:\n"
            f"- Keywords: {ctx['category1']['keywords']}\n"
            f"Category **{ctx['category2']['title']}**:\n"
            f"- Keywords: {ctx['category2']['keywords']}\n"
            f"Formulate the difference in two or three information-rich sentences. Always use the **exact category names** — {ctx['category1']['title']!r} and {ctx['category2']['title']!r}."
            "Explain how the categories differ in terms of content, objects, purpose, target audience, or usage context. "
            "Do not write vague or generic phrases. Do not use constructions like 'the first category' or 'the second category'. "
            "Ignore toponyms and service words (e.g. price/ціна, delivery/доставка, phone/тетефон, photo/фото, used/б/у, new/нове, etc.). "
            "The answer must be information-dense and suitable for machine processing.\n\n"
            "Example answer:\n"
            f"Основна відмінність між {ctx['category1']['title']!r} та {ctx['category2']['title']!r} полягає у тому, що ..."
        )

    def parse_response(self, raw: str) -> dict[str, Any]:
        ...