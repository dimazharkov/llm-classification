import re

# Только буквы, цифры, пробел, дефис и подчёркивание
ALLOWED_CHARS_RE = re.compile(r"[^\w\s\-_]", re.UNICODE)

# Повторяющиеся символы вроде *** или --- (не трогаем одиночные - и _)
REPEATED_NONWORD_RE = re.compile(r"([^\w\s])\1{1,}", re.UNICODE)

# Управляющие символы: \r, \n, \t и прочее
CONTROL_CHARS_RE = re.compile(r"[\r\n\t\f\v]", re.UNICODE)

# Сжимаем пробелы
MULTISPACE_RE = re.compile(r"\s+", re.UNICODE)


def plain_text(text: str) -> str:
    """Нормализует текст объявления:
    - Приводит к нижнему регистру
    - Удаляет управляющие символы (\r, \n, \t и пр.)
    - Удаляет повторы вроде *** или ===
    - Удаляет запрещённые символы, кроме - и _
    - Сжимает пробелы
    """
    text = text.lower()
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = REPEATED_NONWORD_RE.sub(" ", text)
    text = ALLOWED_CHARS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_result(raw_text: str) -> str:
    # Убираем кавычки и апострофы
    text = raw_text.replace('"', "").replace("'", "").strip()

    # Берем только первую строку (до первого перевода строки)
    if "\n" in text:
        text = text.split("\n")[0].strip()

    # Берем только до первой запятой, если есть
    if "," in text:
        text = text.split(",")[0].strip()

    return text.lower()


def clean_html_tags(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    return text.strip()


def fix_latin_letters(text: str) -> str:
    mapping = {
        "a": "а",
        "A": "А",
        "c": "с",
        "C": "С",
        "e": "е",
        "E": "Е",
        "i": "і",
        "I": "І",
        "o": "о",
        "O": "О",
        "p": "р",
        "P": "Р",
        "x": "х",
        "X": "Х",
        "y": "у",
        "Y": "У",
        "b": "в",
        "B": "В",
        "h": "н",
        "H": "Н",
        "k": "к",
        "K": "К",
        "m": "м",
        "M": "М",
        "t": "т",
        "T": "Т",
    }
    for latin, cyr in mapping.items():
        text = text.replace(latin, cyr)
    return text


def normalize_text(text: str) -> str:
    return text.strip().lower()


def clean_text(text: str) -> str:
    return normalize_text(fix_latin_letters(text))
