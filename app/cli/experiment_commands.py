import typer

from app.infrastructure.di import Container

app = typer.Typer()


@app.command()
def one(
    adverts_path: str = typer.Option("/adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("/categories.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("/experiment_one.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: int = typer.Option(1, help="Задержка между запросами"),
) -> None:
    container = Container()
    container.config.ADVERTS_FILE_PATH.from_value(adverts_path)
    container.config.CATEGORIES_FILE_PATH.from_value(categories_path)
    container.config.TARGET_FILE_PATH.from_value(target_path)

    controller = container.experiment_controller()
    use_case = container.experiment_one()
    controller.run(
        use_case,
        num_cases=num_cases,
        rate_limit=rate_limit,
    )


@app.command()
def two(
    adverts_path: str = typer.Option("/adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("/categories.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("/experiment_two.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: int = typer.Option(1, help="Задержка между запросами"),
) -> None:
    container = Container()
    container.config.ADVERTS_FILE_PATH.from_value(adverts_path)
    container.config.CATEGORIES_FILE_PATH.from_value(categories_path)
    container.config.TARGET_FILE_PATH.from_value(target_path)

    controller = container.experiment_controller()
    use_case = container.experiment_two()
    controller.run(
        use_case,
        num_cases=num_cases,
        rate_limit=rate_limit,
    )


@app.command()
def three(
    adverts_path: str = typer.Option("/adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("/categories.json", help="Путь к json c категориями"),
    category_pairs_path: str = typer.Option("/category_pairs.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("/experiment_three.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: int = typer.Option(1, help="Задержка между запросами"),
) -> None:
    container = Container()
    container.config.ADVERTS_FILE_PATH.from_value(adverts_path)
    container.config.CATEGORIES_FILE_PATH.from_value(categories_path)
    container.config.TARGET_FILE_PATH.from_value(target_path)
    container.config.CATEGORIES_PAIR_FILE_PATH.from_value(category_pairs_path)

    controller = container.experiment_controller()
    use_case = container.experiment_three(rate_limit=rate_limit)
    controller.run(use_case, num_cases=num_cases, rate_limit=rate_limit)
