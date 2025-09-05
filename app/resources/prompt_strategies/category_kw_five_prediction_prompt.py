category_kw_five_prediction_prompt = (
    "You are an advertisement classifier. "
    "Analyze the advertisement text and select the three most suitable categories from the list.\n\n"
    "Advertisement:\n"
    '"{advert_title} {advert_text}"\n\n'
    "List of categories with keywords:\n"
    "{categories_with_keywords}\n\n"
    "Each category is provided in the format:\n"
    "- category: keyword1, keyword2, ...\n\n"
    "Answer strictly with a list of three categories, one per line, without numbering, quotes, dashes, keywords, formatting, or explanations.\n\n"
)

category_kw_five_prediction_prompt2 = (
    "Ты — классификатор объявлений. "
    "Проанализируй текст объявления и выбери три наиболее подходящие категории из списка.\n\n"
    'Объявление:\n"{advert_title} {advert_text}"\n\n'
    "Список категорий с ключевыми словами:\n"
    "{categories_with_keywords}\n\n"
    "Каждая категория указана в формате:\n"
    "- категория: ключевое_слово1, ключевое_слово2, ...\n\n"
    "Ответь строго списком из трех категорий, по одной в строке, без нумерации, кавычек, тире, ключевых слов, форматирования или пояснений.\n\n"
    "Пример ответа:\n"
    "косметика\n"
    "средства по уходу\n"
    "мебель для кухни"
)

category_kw_five_prediction_prompt1 = (
    "Ты — классификатор объявлений. "
    "Проанализируй текст объявления и выбери пять наиболее подходящих категорий из списка.\n\n"
    'Объявление:\n"{advert_title} {advert_text}"\n\n'
    "Список категорий с ключевыми словами:\n"
    "{categories_with_keywords}\n\n"
    "Каждая категория указана в формате:\n"
    "- категория: ключевое_слово1, ключевое_слово2, ...\n\n"
    "Ответь строго списком из пяти категорий, по одной в строке, без нумерации, кавычек, тире, ключевых слов, форматирования или пояснений.\n\n"
    "Пример ответа:\n"
    "аренда офисов\n"
    "недвижимость\n"
    "коммерческая аренда\n"
    "офисные помещения\n"
    "бизнес центр"
)
