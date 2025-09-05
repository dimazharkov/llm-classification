category_prediction_prompt = (
    "You are an advertisement classifier. "
    "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
    "Advertisement:\n"
    '"{advert_text}"\n\n'
    "Categories:\n{category_titles}\n\n"
    "Answer strictly with the category name only."
)

category_prediction_prompt3 = (
    "You are an advertisement classifier. "
    "Analyze the advertisement text and choose the most appropriate category from the list.\n\n"
    "Advertisement:\n"
    '"{advert_text}"\n\n'
    "List of categories:\n{category_titles}\n\n"
    "Answer strictly with the category name only."
)

category_prediction_prompt2 = (
    "Ты — классификатор объявлений. "
    "Проанализируй текст объявления и выбери наиболее подходящую категорию из списка.\n\n"
    'Объявление:\n"{advert_title} {advert_text}"\n\n'
    "Список категорий:\n{category_titles}\n\n"
    "Ответь строго в формате: название_категории, уверенность\n"
    "Где:\n"
    "- название_категории — одна из категорий из списка;\n"
    "- уверенность — число от 0 до 1 с двумя знаками после запятой (например, 0.85).\n\n"
    "Пример ответа: аренда офисов, 0.92\n"
    "Не добавляй пояснений, кавычек, форматирования, тегов или лишнего текста."
)

category_prediction_prompt1 = (
    "Ты — классификатор объявлений. "
    "Прочитай текст объявления ниже и выбери наиболее подходящую категорию из предложенного списка.\n\n"
    'Объявление:\n"{advert_title} {advert_text}"\n\n'
    "Категории:\n{category_titles}\n\n"
    "Ответь **только** названием одной категории **без пояснений**, **без кавычек** и **без форматирования**."
)
