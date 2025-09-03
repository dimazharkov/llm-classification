from typing import Any

from core.contracts.prompt_strategy import PromptStrategy


class AdvertSummarize(PromptStrategy[str]):
    name = "advert_summarize"

    def build_prompt(self, ctx: dict[str, Any]) -> str:
        return (
            f"Write a short summary of the advertisement: {ctx['title']!r} {ctx['text']!r}.\n\n"
            "The summary must contain **no more than 15 meaningful words** that capture the essence of the ad. "
            "If fewer words are enough, use fewer — do not add filler.\n"
            "Do NOT include:\n"
            "- prepositions, conjunctions, or stopwords,\n"
            "- numbers or units of measurement (e.g. m, sq.m, usd, pcs, м, кв.м, грн, шт, etc.),\n"
            "- capital letters — all words must be lowercase,\n"
            "- punctuation, HTML, quotation marks, or explanations,\n"
            "- line breaks — the output must be a single line,\n"
            "- place names (cities, countries, streets),\n"
            "- words like: used, price, phone, delivery, new, photo, or their synonyms.\n\n"
            "Return only the words separated by spaces, with no period at the end and no line breaks."
        )

    def parse_response(self, raw: str) -> str:
        ...
