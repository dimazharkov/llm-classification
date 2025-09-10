from typing import Any

from src.core.contracts.advert_repository import AdvertRepository
from src.core.contracts.category_repository import CategoryRepository
from src.core.contracts.file_repository import FileRepository
from src.core.use_cases.build_category_bow import BuildCategoryBowUseCase


class CategoryService:
    def __init__(
        self,
        category_repo: CategoryRepository,
        advert_repo: AdvertRepository,
        file_repo: FileRepository[dict[str, Any]],
    ) -> None:
        self.category_repo = category_repo
        self.advert_repo = advert_repo
        self.file_repo = file_repo

    def build_bow(self, top_k: int = 20) -> None:
        categories = self.category_repo.get_all()
        adverts = self.advert_repo.get_all()

        use_case = BuildCategoryBowUseCase(categories)
        categories_with_bow = use_case.run(adverts, top_k)

        self.file_repo.save_list(categories_with_bow)
