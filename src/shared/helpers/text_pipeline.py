from collections.abc import Callable

from src.shared.helpers.text_utils import normalize_text, replace_latin_lookalikes


def text_pipline(text: str, *functions: Callable[[str], str]) -> str:
    for fn in functions:
        text = fn(text)
    return text


def normalize_and_fix_latin_pipeline(text: str) -> str:
    return text_pipline(text, replace_latin_lookalikes, normalize_text)
