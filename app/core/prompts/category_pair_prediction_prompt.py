category_pair_prediction_prompt = (
    "Ты — эксперт по классификации объявлений. "
    "Тебе дано объявление, а также две категории с описанием различий между ними.\n\n"
    "Объявление:\n"
    '"{advert.advert_title} {advert.advert_text}"\n\n'
    "Категория 1: {category1.title}\n"
    "Категория 2: {category2.title}\n\n"
    "Описание различий между категориями:\n"
    "{difference}\n\n"
    "Выбери, к какой из двух категорий относится объявление. "
    'Если объявление не соответствует ни одной из них, выбери вариант "другое".\n\n'
    "Ответь строго в формате:\n"
    "<название_категории>, <уверенность>\n\n"
    "Где:\n"
    '- <название_категории> — либо "{category1.title}", либо "{category2.title}", либо "другое";\n'
    "- <уверенность> — число от 0 до 1 с двумя знаками после запятой (например, 0.87).\n\n"
    "Примеры ответов:\n"
    "{category1.title}, 0.73\n"
    "другое, 0.62\n\n"
    "Не добавляй никаких пояснений, форматирования или текста, кроме указанного формата."
)

category_pair_prediction_prompt1 = (
    "Ты — эксперт по классификации объявлений. "
    "Тебе дано объявление, а также две категории с описанием различий между ними.\n\n"
    "Объявление:\n"
    '"{advert.advert_title} {advert.advert_text}"\n\n'
    "Категория 1: {category1.title}\n"
    "Категория 2: {category2.title}\n\n"
    "Описание различий между категориями:\n"
    "{difference}\n\n"
    "Выбери, к какой из двух категорий относится объявление. "
    "Ответь строго в формате:\n"
    "<название_категории>, <уверенность>\n\n"
    "Где:\n"
    '- <название_категории> — либо "{category1.title}" либо "{category2.title}";\n'
    "- <уверенность> — число от 0 до 1 с двумя знаками после запятой (например, 0.87).\n\n"
    "Пример ответа:\n"
    "{category1.title}, 0.73\n\n"
    "Не добавляй никаких пояснений, форматирования или текста, кроме указанного формата."
)
