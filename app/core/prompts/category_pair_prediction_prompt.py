category_pair_prediction_prompt = (
    "You are an expert in advertisement classification. "
    "You are given an advertisement, two categories, and a description of their differences.\n\n"
    "Advertisement:\n"
    "\"{advert.advert_summary}\"\n\n"
    "Categories:\n"
    "- {category1.title}: {category1_keywords}\n"
    "- {category2.title}: {category2_keywords}\n\n"
    "Description of differences:\n{difference}\n\n"
    "Rules:\n"
    "- Choose exactly one label: «{category1.title}», «{category2.title}», or «other».\n"
    "- Match the meaning of the advertisement strictly with the provided features.\n"
    "- Ignore brands, condition, price/ціна, delivery, phone numbers, toponyms, HTML, and irrelevant words.\n"
    "- If the type of object/product/service, its purpose, or its context does not fit either category — answer «other».\n"
    "- If the match is weak, contradictory, or only general words overlap — answer «other».\n"
    "- Internally assess the match with each category (0–1); if both scores are low (<0.6) — answer «other». Do not output the scores.\n"
    "- Always use the exact category names; never write “first/second category”.\n\n"
    "Answer strictly with a single lowercase word without quotes: «{category1.title}» or «{category2.title}» or «other»."
)

category_pair_prediction_prompt6 = (
    "You are an expert in advertisement classification. You are given an advertisement, two categories, and a description of their differences.\n"
    "Task — choose exactly one label from: «{category1.title}», «{category2.title}», or «other».\n\n"
    "Description of differences:\n{difference}\n\n"
    "Rules (no examples):\n"
    "- Match the meaning of the advertisement strictly with the features and boundaries from the description.\n"
    "- Ignore brands, condition, price/ціна, delivery, phone numbers, toponyms, HTML, and irrelevant words.\n"
    "- If the type of object/product/service, its purpose, or its context does not fit either category — answer «other».\n"
    "- If the match is weak, contradictory, or only general words overlap — answer «other».\n"
    "- Internally assess the match with each category (0–1); if both scores are low (<0.6) — answer «other». Do not output the scores.\n"
    "- Always use the exact category names «{category1.title}» and «{category2.title}»; never write “first/second category”.\n\n"
    "Advertisement:\n"
    "\"{advert.advert_summary}\"\n\n"
    "Answer strictly with a single lowercase word without quotes: «{category1.title}» or «{category2.title}» or «other»."
)

# "Category “{category1.title}” is characterized by keywords: {category1_keywords}.\n"
# "Category “{category2.title}” is characterized by keywords: {category2_keywords}.\n\n"

category_pair_prediction_prompt5 = (
    "Ты — эксперт по классификации объявлений. Тебе дано объявление, две категории и описание их различий.\n"
    "Задача — выбрать ровно одну метку из: «{category1.title}», «{category2.title}», «другое».\n\n"
    "Описание различий:\n"
    "Категория «{category1.title}» характеризуется ключевыми словами: {category1_keywords}.\n"
    "Категория «{category2.title}» характеризуется ключевыми словами: {category2_keywords}.\n\n"
    "Правила (без примеров):\n"
    "- Сопоставляй смысл объявления строго с признаками и границами из описания различий.\n"
    "- Игнорируй бренды, состояние, цену/ціну, доставку, телефоны, топонимы, HTML, мусорные слова.\n"
    '- Если тип объекта/товара/услуги, назначение или контекст не подпадают ни под одну категорию — ответ "другое".\n'
    '- Если совпадение слабое, признаки противоречивы, совпадают только общие слова — ответ "другое".\n'
    '- Внутренне оцени соответствие каждой категории (0–1); если обе оценки низкие (<0.6) — ответ "другое". Оценки не выводи.\n'
    "- Используй точные названия категорий «{category1.title}» и «{category2.title}»; не пиши «первая/вторая категория».\n\n"
    'Объявление:\n"{advert.advert_summary}"\n\n'
    "Ответь строго одним словом в нижнем регистре без кавычек: {category1.title} или {category2.title} или другое."
)

category_pair_prediction_prompt4 = (
    "Ты — эксперт по классификации объявлений. Тебе дано объявление, две категории и описание их различий.\n"
    'Задача — выбрать ровно одну метку из: "{category1.title}", "{category2.title}", "другое".\n\n'
    "Описание различий:\n{difference}\n\n"
    "Правила (без примеров):\n"
    "- Сопоставляй смысл объявления строго с признаками и границами из описания различий.\n"
    "- Игнорируй бренды, состояние, цену/ціну, доставку, телефоны, топонимы, HTML, мусорные слова.\n"
    '- Если тип объекта/товара/услуги, назначение или контекст не подпадают ни под одну категорию — ответ "другое".\n'
    '- Если совпадение слабое, признаки противоречивы, совпадают только общие слова — ответ "другое".\n'
    '- Внутренне оцени соответствие каждой категории (0–1); если обе оценки низкие (<0.6) — ответ "другое". Оценки не выводи.\n'
    "- Используй точные названия категорий «{category1.title}» и «{category2.title}»; не пиши «первая/вторая категория».\n\n"
    'Объявление:\n"{advert.advert_summary}"\n\n'
    "Ответь строго одним словом в нижнем регистре без кавычек: {category1.title} или {category2.title} или другое."
)

category_pair_prediction_prompt3 = (
    "Ты — эксперт по классификации объявлений. "
    "Тебе дано объявление, а также две категории с описанием различий между ними.\n"
    "Твоя задача — определить, к какой из двух категорий относится объявление.\n"
    'Если объявление не подходит ни под одну из категорий — выбери "другое". Это правильный и желаемый ответ в таких случаях.\n\n'
    "Объявление:\n"
    '"{advert.advert_title} {advert.advert_text}"\n'
    "Категория 1: {category1.title}\n"
    "Категория 2: {category2.title}\n\n"
    "Описание различий между категориями:\n"
    "{difference}\n\n"
    'Обрати внимание: если объявление не соответствует ни одной из категорий, выбери "другое". Это не ошибка. Это нормальный и желаемый вариант, если объявление не совпадает по смыслу ни с одной из представленных категорий.\n\n'
    "Ответь строго в формате:\n"
    "<название_категории>\n"
    'Где: <название_категории> — либо "{category1.title}", либо "{category2.title}", либо "другое".\n'
    "Не добавляй никаких пояснений, форматирования или текста, кроме указанного формата."
)

category_pair_prediction_prompt2 = (
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
