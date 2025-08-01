from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.helpers.prompt_helper import format_prompt
from app.core.prompts.category_kw_five_prediction_prompt import category_kw_five_prediction_prompt
from app.repositories.category_file_repository import CategoryFileRepository
from app.shared.helpers.text_pipeline import normalize_and_fix_latin_pipeline


class PredictFiveCategoriesUseCase:
    def __init__(self, llm: LLMClientContract, category_repo: CategoryFileRepository):
        self.llm = llm
        self.category_repo = category_repo
        self.categories_with_keywords = self._get_categories_with_keywords()

    def run(self, advert: Advert) -> list[Category]:
        prompt = format_prompt(
            category_kw_five_prediction_prompt,
            advert_title=advert.advert_title,
            advert_text=advert.advert_text,
            categories_with_keywords=self.categories_with_keywords,
        )
        model_result = self.llm.generate(prompt)
        predicted_categories = self._parse_top_categories(model_result)

        if advert.category_title in predicted_categories:
            print(f"[!] advert category among 5 categories: {advert.category_title}")
        else:
            print(f"advert_category={advert.category_title}, predicted_categories={predicted_categories}")

        return self._prep_output(predicted_categories)

    def _get_categories_with_keywords(self) -> str:
        return "\n".join(f"- {c.title}: {', '.join(c.bow or [])}" for c in self.category_repo.get())

    def _parse_top_categories(self, model_result: str) -> list[str]:
        lines = [line.strip("- ") for line in model_result.strip().split("\n")]
        top_categories = [normalize_and_fix_latin_pipeline(line) for line in lines if line]
        return top_categories[:5]

    def _prep_output(self, predicted_categories) -> list[Category]:
        return [category for category in self.category_repo.get() if category.title in predicted_categories]
