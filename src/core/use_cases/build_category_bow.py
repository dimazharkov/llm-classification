import re
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.domain.advert import Advert
from src.core.domain.category import Category


class BuildCategoryBowUseCase:
    def __init__(
        self,
        categories: list[Category],
        *,
        stop_words: set[str] | None = None,
        min_df: float = 2,  # under the volume of ads
        max_df: float = 0.7,
        token_pattern: str = r"(?u)\b(?!\d)\w[\w\-]+\b",  # noqa: S107
    ) -> None:
        self.categories = categories
        # base stop-list
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
            "б/у",
        }
        self.min_df = min_df
        self.max_df = max_df
        self.token_pattern = token_pattern
        self._token_re = re.compile(self.token_pattern)

    def run(self, adverts: list[Advert], top_k: int = 25) -> list[Category]:
        # BoW: Collect words by categories with the same token normalization
        category_words = defaultdict(list)
        for advert in adverts:
            s = (advert.advert_summary or "").strip().lower()
            if not s:
                continue
            # split() collapses any newlines/tabs/multi-spaces
            tokens = s.split()
            tokens = self._filter_tokens(tokens)
            category_words[advert.category_id].extend(tokens)

        bow_per_category = self._get_bow_per_category(category_words, top_k=top_k)

        # TF-IDF by ads with subsequent aggregation to the category level
        tf_idf_per_category = self._get_tf_idf_per_category_from_adverts(adverts, top_k=top_k)

        categories_with_bow: list[Category] = []
        for category in self.categories:
            bow = bow_per_category.get(category.id, [])
            tf_idf = tf_idf_per_category.get(category.id, [])
            categories_with_bow.append(Category(id=category.id, title=category.title, bow=bow, tf_idf=tf_idf))
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
            # normalize spaces; no need to remove stop words here - vectorizer will do it
            s = " ".join(s.split())
            docs.append(s)
            cat_ids.append(a.category_id)

        # we want to cover all categories from self.categories, even if they don't have any ads
        uniq_cats = [c.id for c in self.categories]
        if not docs:
            return {cid: [] for cid in uniq_cats}

        cat_to_idx = {cid: i for i, cid in enumerate(uniq_cats)}
        # ignore ads with categories that are not in self.categories
        cat_idx = np.array([cat_to_idx[cid] for cid in cat_ids if cid in cat_to_idx], dtype=np.int32)
        # filter the relevant ads
        if len(cat_idx) != len(cat_ids):
            filtered_docs = []
            for d, cid in zip(docs, cat_ids, strict=False):
                if cid in cat_to_idx:
                    filtered_docs.append(d)
            docs = filtered_docs
        if not docs:
            return {cid: [] for cid in uniq_cats}

        n = len(docs)
        if n < 20_000:
            self.min_df = 2
        elif n < 100_000:
            self.min_df = 5
        else:
            self.min_df = 0.005

        vectorizer = TfidfVectorizer(
            lowercase=False,  # already led to lower
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=self.token_pattern,
            stop_words=list(self.stop_words) if self.stop_words else None,  # единый стоп-лист
        )
        x = vectorizer.fit_transform(docs)  # [n_docs, n_terms]
        feats = np.array(vectorizer.get_feature_names_out())

        # aggregate: sum of tf-idf for ads of each category
        x_coo = x.tocoo(copy=False)
        # cat_idx should match X_coo.row; recalculate to be sure
        # build a parallel array of category indices for each row of X
        doc_cat_idx = np.array([cat_to_idx[cid] for cid in cat_ids if cid in cat_to_idx], dtype=np.int32)
        cat_rows = doc_cat_idx[x_coo.row]

        c = len(uniq_cats)
        m = sparse.coo_matrix(
            (x_coo.data, (cat_rows, x_coo.col)),
            shape=(c, x.shape[1]),
            dtype=x.dtype,
        ).tocsr()

        # average TF-IDF for category ads (not the sum) to compare categories with different "load"
        counts = np.bincount(doc_cat_idx, minlength=c).astype(np.float32)
        counts[counts == 0] = 1.0
        m_mean = m.multiply(1.0 / counts[:, None])

        # top-k for each category without toarray()
        tf_idf_per_category: dict[int, list[str]] = {}
        for i, cid in enumerate(uniq_cats):
            row = m_mean.getrow(i)
            if row.nnz == 0:
                tf_idf_per_category[cid] = []
                continue
            order = np.argsort(row.data)[::-1][:top_k]
            top_cols = row.indices[order]
            tf_idf_per_category[cid] = feats[top_cols].tolist()
        return tf_idf_per_category
