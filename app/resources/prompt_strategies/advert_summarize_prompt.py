advert_summarize_prompt = (
    'Write a short summary of the advertisement: "{title}" {text}.\n\n'
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

advert_summarize_prompt3 = (
    'Сделай краткое резюме объявления: "{title}" {text}.\n\n'
    "Результат должен содержать **не более 15** **значимых слов**, описывающих суть объявления. "
    "Если значимых слов меньше — используй меньше, без добавления мусора.\n"
    "Не используй:\n"
    "- предлоги, союзы и служебные слова,\n"
    "- цифры, числа, единицы измерения (например: м, кв.м, грн, шт и т.п.),\n"
    "- заглавные буквы — все слова должны быть в нижнем регистре,\n"
    "- знаки препинания, HTML, кавычки или пояснения,\n"
    "- переводы строк — всё должно быть в одной строке,\n"
    "- топонимы (названия городов, стран, улиц),\n"
    "- слова: б/у, цена, ціна, телефон, доставка, нове, фото и их синонимы.\n\n"
    "Верни только слова, разделённые пробелами. Без точки в конце и перевода строки."
)

advert_summarize_prompt2 = (
    'Сделай краткое резюме объявления: "{title}" {text}.\n\n'
    "Результат должен содержать **ровно 10** **значимых слов**, описывающих суть объявления.\n"
    "Не используй:\n"
    "- предлоги, союзы и служебные слова,\n"
    "- цифры, числа, единицы измерения (например: м, кв.м, грн, шт и т.п.),\n"
    "- заглавные буквы — все слова должны быть в нижнем регистре,\n"
    "- знаки препинания, HTML, кавычки или пояснения,\n"
    "- переводы строк — всё должно быть в одной строке.\n\n"
    "Верни только слова, разделённые пробелами. Без точки в конце."
)

advert_summarize_prompt_1 = (
    'Сделай краткое резюме объявления: "{title}" {text}.\n\n'
    "Ответ должен содержать ровно 10 значимых слов, описывающих суть объявления.\n"
    "Не используй предлоги, союзы и другие частотные или служебные слова.\n"
    "Не добавляй пунктуацию, цифры, HTML или пояснения.\n"
    "Ответ должен быть одной строкой, слова разделяй пробелами."
)
