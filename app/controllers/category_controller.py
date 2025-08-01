from collections.abc import Iterable

from app.core.domain.advert import Advert
from app.core.domain.category import Category
from app.core.dto.category_raw import CategoryRaw
from app.core.use_cases.build_category_bow import BuildCategoryBowUseCase
from app.core.use_cases.preprocess_categories import PreprocessCategoriesUseCase
from app.helpers.os_helper import load_from_disc, save_to_disc


class CategoryController:
    def preprocess(self, source_path: str, target_path: str, selected_ids: Iterable[int] = None) -> None:
        raw = load_from_disc(source_path)
        parsed: dict[int, CategoryRaw] = {int(cid): CategoryRaw.model_validate(cat) for cid, cat in raw.items()}

        processed = PreprocessCategoriesUseCase(parsed).run(selected_ids)

        payload = [category.model_dump(mode="json") for category in processed]
        save_to_disc(payload, target_path)

    def build_bow(self, categories_path: str, adverts_path: str, target_path: str, top_k: int = 20) -> None:
        raw_categories = load_from_disc(categories_path)
        categories = [Category.model_validate(category) for category in raw_categories]

        raw_adverts = load_from_disc(adverts_path)
        adverts = [Advert.model_validate(advert) for advert in raw_adverts]

        categories_with_bow = BuildCategoryBowUseCase(categories).run(adverts, top_k)

        payload = [category_with_bow.model_dump(mode="json") for category_with_bow in categories_with_bow]
        save_to_disc(payload, target_path)
