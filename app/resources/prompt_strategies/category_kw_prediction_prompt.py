category_kw_prediction_prompt = (
    "You are an advertisement classifier. "
    "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
    "Advertisement:\n"
    "\"{advert_text}\"\n\n"
    "Categories with keywords:\n"
    "{categories_with_keywords}\n\n"
    "Each category is provided in the format:\n"
    "- category: keyword1, keyword2, ...\n\n"
    "Answer strictly with the category name only."
)

category_kw_prediction_prompt2 = (
    "You are an advertisement classifier. "
    "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
    "Advertisement:\n"
    "\"{advert_text}\"\n\n"
    "List of categories with keywords:\n"
    "{categories_with_keywords}\n\n"
    "Each category is provided in the format:\n"
    "- category: keyword1, keyword2, ...\n\n"
    "Answer strictly with the category name only. "
)

category_kw_prediction_prompt1 = (
    "Ты — классификатор объявлений. "
    "Проанализируй текст объявления и выбери наиболее подходящую категорию из списка.\n\n"
    'Объявление:\n"{advert_title} {advert_text}"\n\n'
    "Список категорий с ключевыми словами:\n"
    "{categories_with_keywords}\n\n"
    "Каждая категория указана в формате:\n"
    "- категория: ключевое_слово1, ключевое_слово2, ...\n\n"
    "Ответь строго в формате: название_категории, уверенность\n"
    "Где:\n"
    "- название_категории — одна из категорий из списка;\n"
    "- уверенность — число от 0 до 1 с двумя знаками после запятой (например, 0.85).\n\n"
    "Пример ответа: аренда офисов, 0.92\n"
    "Не добавляй пояснений, кавычек, форматирования, тегов или лишнего текста."
)
