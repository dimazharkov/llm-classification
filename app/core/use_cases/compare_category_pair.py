import itertools
import time
from dataclasses import dataclass

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.dto.category_diff import CategoryDiff
from app.core.helpers.prompt_helper import format_prompt
from app.core.prompts.category_difference_prompt import category_difference_prompt
from app.repositories.advert_file_repository import AdvertFileRepository
from app.repositories.category_pair_diff_repository import CategoryPairDiffRepository
from app.repositories.category_pair_file_repository import CategoryPairFileRepository


@dataclass(slots=True)
class CategoryData:
    title: str
    keywords: str
    examples: str | None = None


class CompareCategoryPairUseCase:
    def __init__(
        self,
        llm: LLMClientContract,
        advert_repo: AdvertFileRepository,
        category_pair_repo: CategoryPairFileRepository,
    ):
        self.llm = llm
        self.advert_repo = advert_repo
        self.category_pair_repo = category_pair_repo

    def run(self, category_list: list[Category], rate_limit: float = 0.5) -> CategoryPairDiffRepository:
        if not category_list:
            print("Achtung! category_list is empty!")

        category_pair_diff = CategoryPairDiffRepository()
        for category1, category2 in itertools.combinations(category_list, 2):
            pair_key = (category1.id, category2.id)
            diff_text = self.category_pair_repo.get(pair_key)
            if not diff_text:
                print(f"[{category1.id}:{category2.id}] diff isn't found, preparing ...")
                diff_text = self._prep_categories_diff(category1, category2)
                self.category_pair_repo.add(pair_key, diff_text)
                time.sleep(rate_limit)
            else:
                print(f"[{category1.id}:{category2.id}] ok")

            category_pair_diff.add(
                pair_key,
                CategoryDiff(category1=category1, category2=category2, difference=diff_text),
            )

        self.category_pair_repo.save()
        # print(f"category_pair_diff: {category_pair_diff}")
        return category_pair_diff

    def _prep_categories_diff(self, category1: Category, category2: Category) -> str:
        category_data1 = self._get_category_data(category1)
        category_data2 = self._get_category_data(category2)

        prompt = format_prompt(
            category_difference_prompt,
            category1=category_data1,
            category2=category_data2,
        )
        # print(f"category_difference_prompt:\n {prompt}")
        category_diff = self.llm.generate(prompt)
        # print(f"category_difference_prompt result:\n {category_diff}")
        return category_diff

    def _get_category_data(self, category: Category) -> CategoryData:
        # adverts_by_category = self.advert_repo.get_adverts_by_category(category)

        return CategoryData(
            title=category.title,
            keywords=", ".join(f"{kw}" for kw in category.tf_idf or []),
            # examples=self._get_adverts_expamples(adverts_by_category),
        )

    def _get_adverts_expamples(self, adverts: list[Advert], limit: int = 5) -> str:
        return "\n".join(f"- {advert.advert_text}" for advert in adverts[:limit])
