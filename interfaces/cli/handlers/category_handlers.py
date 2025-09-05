from app.services.category_service import CategoryService


def category_preprocess(source_path: str, target_path: str, selected_ids: list[int]) -> None:
    service = CategoryService()
    service.preprocess(source_path, target_path, selected_ids)


def category_enrich(categories_path: str, adverts_path: str, target_path: str, top_k: int) -> None:
    service = CategoryService()
    service.build_bow(categories_path, adverts_path, target_path, top_k)


def category_compare(adverts_path: str, categories_path: str, category_pairs_path: str) -> None:
    service = CategoryService()
    service.compare_category_pair(adverts_path, categories_path, category_pairs_path)
