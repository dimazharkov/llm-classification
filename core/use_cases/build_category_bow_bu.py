from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from core.contracts.use_case_contract import UseCaseContract
from core.domain.advert import Advert
from core.domain.category import Category


class BuildCategoryBowUseCase(UseCaseContract):
    def __init__(self, categories: list[Category]):
        self.categories = categories

    def run(self, adverts: list[Advert], top_k: int = 20) -> list[Category]:
        category_words = defaultdict(list)

        for advert in adverts:
            if not advert.advert_summary:
                print("advert has no advert_summary")
                continue
            # words = advert.advert_summary.lower().split()
            words = advert.advert_summary.lower().replace("\n", "").split()
            category_words[advert.category_id].extend(words)

        bow_per_category = self._get_bow_per_category(category_words, top_k=top_k)

        tf_idf_per_category = self._get_tf_idf_per_category(category_words, top_k=top_k)

        categories_with_bow = []
        for category in self.categories:
            bow = bow_per_category.get(category.id, [])
            tf_idf = tf_idf_per_category.get(category.id, [])
            category_with_bow = category.model_copy(update={"bow": bow, "tf_idf": tf_idf})
            categories_with_bow.append(category_with_bow)

        return categories_with_bow

    def _get_bow_per_category(self, category_words: dict[int, list[str]], top_k: int) -> dict[int, list[str]]:
        bow_per_category = {}
        for category_id, words in category_words.items():
            if not words:
                bow_per_category[category_id] = []
                continue
            freq = Counter(words)
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            bow_per_category[category_id] = [word for word, _ in sorted_words[:top_k]]
        return bow_per_category

    def _get_tf_idf_per_category(self, category_words: dict[int, list[str]], top_k: int) -> dict[int, list[str]]:
        documents = [" ".join(words) for words in category_words.values()]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        tf_idf_per_category = {}

        for idx, category_id in enumerate(category_words.keys()):
            row = tfidf_matrix[idx]
            tfidf_scores = zip(feature_names, row.toarray()[0], strict=False)

            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

            top_words = [word for word, score in sorted_scores[:top_k]]
            tf_idf_per_category[category_id] = top_words

        return tf_idf_per_category
