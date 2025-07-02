from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.domain.advert import Advert
from app.core.domain.category import Category


class BuildCategoryBowUseCase:
    def __init__(self, categories: list[Category]):
        self.categories = categories

    def run(self, adverts: list[Advert], top_k: int = 20) -> list[Category]:
        category_words = defaultdict(list)

        for advert in adverts:
            if not advert.advert_summary:
                continue
            words = advert.advert_summary.lower().split()
            category_words[advert.category_id].extend(words)

        bow_per_category = self._get_bow_per_category(
            category_words
        )

        tf_idf_per_category = self._get_tf_idf_per_category(
            category_words
        )

        categories_with_bow = []
        for category in self.categories:
            bow = bow_per_category.get(category.id, [])
            tf_idf = tf_idf_per_category.get(category.id, [])
            category_with_bow = category.model_copy(
                update={
                    "bow": bow,
                    "tf_idf": tf_idf
                }
            )
            categories_with_bow.append(category_with_bow)

        return categories_with_bow

    def _get_bow_per_category(self, category_words: dict[int, list[str]]) -> dict[int, list[str]]:
        bow_per_category = {}
        for category_id, words in category_words.items():
            # подсчет частот
            freq = Counter(words)
            # получаем список (word, count), сортируем по count по убыванию
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

            # проверяем, все ли слова имеют одинаковую частоту
            counts = [count for word, count in sorted_words]
            if len(set(counts)) == 1:
                # все слова имеют одинаковую частоту, берем первые 20
                top_words = [word for word, count in sorted_words[:20]]
            else:
                # берем только 20 самых частотных
                top_words = [word for word, count in sorted_words[:20]]

            bow_per_category[category_id] = top_words

        return bow_per_category

    def _get_tf_idf_per_category(self, category_words: dict[int, list[str]]) -> dict[int, list[str]]:
        documents = [' '.join(words) for words in category_words.values()]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        feature_names = vectorizer.get_feature_names_out()

        tf_idf_per_category = {}

        # для каждой категории
        for idx, category_id in enumerate(category_words.keys()):
            # получаем TF-IDF для текущего документа
            row = tfidf_matrix[idx]
            tfidf_scores = zip(feature_names, row.toarray()[0])

            # сортируем по TF-IDF
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

            # выбираем топ-20
            top_words = [word for word, score in sorted_scores[:20]]
            tf_idf_per_category[category_id] = top_words

        return tf_idf_per_category