import typer

from interfaces.cli.handlers.experiment_handlers import experiment_one, experiment_two, experiment_three

app = typer.Typer()


@app.command()
def one(
    adverts_path: str = typer.Option("adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("categories.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("experiment_one.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: float = typer.Option(0.5, help="Задержка между запросами"),
) -> None:
    experiment_one(adverts_path, categories_path, target_path, num_cases, rate_limit)


@app.command()
def two(
    adverts_path: str = typer.Option("adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("categories.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("experiment_two.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: float = typer.Option(0.5, help="Задержка между запросами"),
) -> None:
    experiment_two(adverts_path, categories_path, target_path, num_cases, rate_limit)


@app.command()
def three(
    adverts_path: str = typer.Option("/adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("/categories.json", help="Путь к json c категориями"),
    category_pairs_path: str = typer.Option("/category_pairs.json", help="Путь к json c категориями"),
    target_path: str = typer.Option("/experiment_three.json", help="Путь для сохранения"),
    num_cases: int = typer.Option(1000, help="Количество экспериментов"),
    rate_limit: float = typer.Option(0.5, help="Задержка между запросами"),
) -> None:
    experiment_three(
        adverts_path, categories_path, category_pairs_path, target_path, num_cases, rate_limit
    )
