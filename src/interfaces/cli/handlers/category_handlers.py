from src.infra.di import Container


def category_prep_bow(categories_path: str, adverts_path: str, target_path: str, top_k: int) -> None:
    container = Container()
    container.config.update(
        {
            "categories_path": categories_path,
            "adverts_path": adverts_path,
            "file_path": target_path,
        },
    )

    service = container.category_service()
    service.build_bow(top_k)
