from infra.di import Container


def experiment_one(adverts_path: str, categories_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
    container = Container()
    container.config.llm_client_name.from_value('gemini')
    container.config.adverts_path.from_value(adverts_path)
    container.config.categories_path.from_value(categories_path)
    container.config.target_path.from_value(target_path)

    service = container.experiment_service()
    use_case = container.experiment_one()

    service.run(
        use_case,
        num_cases=num_cases,
        rate_limit=rate_limit
    )

def experiment_two(adverts_path: str, categories_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
    container = Container()
    container.config.update({
        "llm_client_name": "gemini",
        "adverts_path": adverts_path,
        "categories_path": categories_path,
        "target_path": target_path
    })

    service = container.experiment_service()
    use_case = container.experiment_two()

    service.run(
        use_case,
        num_cases=num_cases,
        rate_limit=rate_limit
    )
#
# def experiment_three(adverts_path: str, categories_path: str, category_pairs_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
#     container = Container()
#     container.config.ADVERTS_FILE_PATH.from_value(adverts_path)
#     container.config.CATEGORIES_FILE_PATH.from_value(categories_path)
#     container.config.TARGET_FILE_PATH.from_value(target_path)
#     container.config.CATEGORIES_PAIR_FILE_PATH.from_value(category_pairs_path)
#
#     controller = container.experiment_controller()
#     use_case = container.experiment_three(rate_limit=rate_limit)
#     controller.run(use_case, num_cases=num_cases, rate_limit=rate_limit)
#
