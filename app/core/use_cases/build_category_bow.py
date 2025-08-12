import re
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.contracts.use_case_contract import UseCaseContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category


class BuildCategoryBowUseCase(UseCaseContract):
    def __init__(
        self,
        categories: list[Category],
        *,
        stop_words: set[str] | None = None,
        min_df: int | float = 2,  # под объём объявлений
        max_df: float = 0.7,
        token_pattern: str = r"(?u)\b(?!\d)\w[\w\-]+\b",
    ):
        self.categories = categories
        # базовый стоп-лист под описание объявлений
        self.stop_words = stop_words or {
            "цена",
            "ціна",
            "телефон",
            "доставка",
            "фото",
            "нове",
            "новый",
            "новий",
            "бу",
            "б",
            "у",
            "б/у",  # на случай заранее токенизированных строк
        }
        self.min_df = min_df
        self.max_df = max_df
        self.token_pattern = token_pattern
        self._token_re = re.compile(self.token_pattern)

    def run(self, adverts: list[Advert], top_k: int = 25) -> list[Category]:
        # BoW: собираем слова по категориям с той же нормализацией токенов
        category_words = defaultdict(list)
        for advert in adverts:
            s = (advert.advert_summary or "").strip().lower()
            if not s:
                print("advert has no advert_summary")
                continue
            # split() схлопывает любые переводы строк/табы/мультипробелы
            tokens = s.split()
            tokens = self._filter_tokens(tokens)
            category_words[advert.category_id].extend(tokens)

        bow_per_category = self._get_bow_per_category(category_words, top_k=top_k)

        # TF-IDF по объявлениям с последующей агрегацией на уровень категорий
        tf_idf_per_category = self._get_tf_idf_per_category_from_adverts(adverts, top_k=top_k)

        categories_with_bow: list[Category] = []
        for category in self.categories:
            bow = bow_per_category.get(category.id, [])
            tf_idf = tf_idf_per_category.get(category.id, [])
            categories_with_bow.append(category.model_copy(update={"bow": bow, "tf_idf": tf_idf}))
        return categories_with_bow

    def _filter_tokens(self, tokens: Iterable[str]) -> list[str]:
        out: list[str] = []
        for t in tokens:
            if not self._token_re.fullmatch(t):
                continue
            if t in self.stop_words:
                continue
            out.append(t)
        return out

    def _get_bow_per_category(self, category_words: dict[int, list[str]], top_k: int) -> dict[int, list[str]]:
        bow_per_category: dict[int, list[str]] = {}
        for category_id, words in category_words.items():
            if not words:
                bow_per_category[category_id] = []
                continue
            freq = Counter(words)
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            bow_per_category[category_id] = [word for word, _ in sorted_words[:top_k]]
        return bow_per_category

    def _get_tf_idf_per_category_from_adverts(self, adverts: list[Advert], top_k: int) -> dict[int, list[str]]:
        docs: list[str] = []
        cat_ids: list[int] = []

        for a in adverts:
            s = (a.advert_summary or "").strip().lower()
            if not s:
                continue
            # нормализуем пробелы; удалять стоп-слова тут не нужно — это сделает vectorizer
            s = " ".join(s.split())
            docs.append(s)
            cat_ids.append(a.category_id)

        # хотим покрыть все категории из self.categories, даже если в них нет объявлений
        uniq_cats = [c.id for c in self.categories]
        if not docs:
            return {cid: [] for cid in uniq_cats}

        cat_to_idx = {cid: i for i, cid in enumerate(uniq_cats)}
        # игнорируем объявления с категориями, которых нет в self.categories
        cat_idx = np.array([cat_to_idx[cid] for cid in cat_ids if cid in cat_to_idx], dtype=np.int32)
        # отфильтруем соответствующие документы
        if len(cat_idx) != len(cat_ids):
            filtered_docs = []
            for d, cid in zip(docs, cat_ids, strict=False):
                if cid in cat_to_idx:
                    filtered_docs.append(d)
            docs = filtered_docs
        if not docs:
            return {cid: [] for cid in uniq_cats}

        N = len(docs)
        if N < 20_000:
            self.min_df = 2
        elif N < 100_000:
            self.min_df = 5
        else:
            self.min_df = 0.005

        vectorizer = TfidfVectorizer(
            lowercase=False,  # уже привели к lower
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=self.token_pattern,
            stop_words=list(self.stop_words) if self.stop_words else None,  # единый стоп-лист
        )
        X = vectorizer.fit_transform(docs)  # [n_docs, n_terms]
        feats = np.array(vectorizer.get_feature_names_out())

        # агрегируем: сумма tf-idf по объявлениям каждой категории
        X_coo = X.tocoo(copy=False)
        # cat_idx должен соответствовать X_coo.row; пересчитаем заново для надёжности
        # строим параллельный массив индексов категорий для каждой строки X
        doc_cat_idx = np.array([cat_to_idx[cid] for cid in cat_ids if cid in cat_to_idx], dtype=np.int32)
        cat_rows = doc_cat_idx[X_coo.row]

        C = len(uniq_cats)
        M = sparse.coo_matrix(
            (X_coo.data, (cat_rows, X_coo.col)),
            shape=(C, X.shape[1]),
            dtype=X.dtype,
        ).tocsr()

        # средний TF-IDF по объявлениям категории (а не сумма), чтобы сравнивать категории разной «нагруженности»
        counts = np.bincount(doc_cat_idx, minlength=C).astype(np.float32)
        counts[counts == 0] = 1.0
        M_mean = M.multiply(1.0 / counts[:, None])

        # top-k по каждой категории без toarray()
        tf_idf_per_category: dict[int, list[str]] = {}
        for i, cid in enumerate(uniq_cats):
            row = M_mean.getrow(i)
            if row.nnz == 0:
                tf_idf_per_category[cid] = []
                continue
            order = np.argsort(row.data)[::-1][:top_k]
            top_cols = row.indices[order]
            tf_idf_per_category[cid] = feats[top_cols].tolist()
        return tf_idf_per_category
