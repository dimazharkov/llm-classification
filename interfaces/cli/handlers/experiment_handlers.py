from infra.di import Container


def experiment_one(adverts_path: str, categories_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
    container = Container()
    container.config.update(
        {
            "llm_client_name": "gemini",
            "adverts_path": adverts_path,
            "categories_path": categories_path,
            "target_path": target_path,
        },
    )

    service = container.experiment_service()
    use_case = container.experiment_one()

    service.run(use_case, num_cases=num_cases, rate_limit=rate_limit)


def experiment_two(adverts_path: str, categories_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
    container = Container()
    container.config.update(
        {
            "llm_client_name": "gemini",
            "adverts_path": adverts_path,
            "categories_path": categories_path,
            "target_path": target_path,
        },
    )

    service = container.experiment_service()
    use_case = container.experiment_two()

    service.run(use_case, num_cases=num_cases, rate_limit=rate_limit)


def experiment_three(adverts_path: str, categories_path: str, categories_pairs_path: str, target_path: str, num_cases: int, rate_limit: float) -> None:
    container = Container()
    container.config.update(
        {
            "llm_client_name": "gemini",
            "adverts_path": adverts_path,
            "categories_path": categories_path,
            "target_path": target_path,
            "categories_pairs_path": categories_pairs_path,
            "rate_limit": rate_limit
        },
    )

    service = container.experiment_service()
    use_case = container.experiment_three()

    service.run(use_case, num_cases=num_cases, rate_limit=rate_limit)
