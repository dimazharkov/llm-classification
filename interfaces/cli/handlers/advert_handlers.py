from app.services.advert_service import AdvertService
from infra.di import Container


def advert_summarize(source_path: str, target_path: str) -> None:
    container = Container()
    llm_client = container.llm_client()
    service = AdvertService()
    service.summarize(
        source_path, target_path, llm_client
    )

def advert_preprocess(source_path: str, target_path: str, per_category: int, category_ids: list[int]) -> None:
    service = AdvertService()
    service.preprocess(
        source_path, target_path, per_category, category_ids
    )

def advert_indexing(source_path: str, target_path: str) -> None:
    service = AdvertService()
    service.indexing(
        source_path, target_path
    )

def advert_analyze(source_path: str, rejected_path: str) -> None:
    service = AdvertService()
    service.analyze_results(
        source_path, rejected_path
    )
