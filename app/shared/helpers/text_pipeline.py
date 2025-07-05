from typing import Callable
from app.shared.helpers.text_utils import replace_latin_lookalikes, normalize_text


def text_pipline(text: str, *functions: list[Callable[[str], str]]) -> str:
    for fn in functions:
        text = fn(text)
    return text

def normalize_and_fix_latin_pipeline(text: str) -> str:
    return text_pipline(
        text,
        replace_latin_lookalikes, normalize_text
    )