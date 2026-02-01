import typer

from src.interfaces.cli.handlers.experiment_handlers import experiment_one, experiment_three, experiment_two

app = typer.Typer()


@app.command()
def one(
    adverts_path: str = typer.Option("adverts.json", help="Path to the adverts json file"),
    categories_path: str = typer.Option("categories.json", help="Path to the categories json file"),
    target_path: str = typer.Option("experiment_one.json", help="Output path"),
    num_cases: int = typer.Option(10_000, help="Number of experiments"),
    delay_s: float = typer.Option(20, help="Delay between requests in seconds"),
) -> None:
    experiment_one(adverts_path, categories_path, target_path, num_cases, delay_s)


@app.command()
def two(
    adverts_path: str = typer.Option("adverts.json", help="Path to the adverts json file"),
    categories_path: str = typer.Option("categories.json", help="Path to the categories json file"),
    target_path: str = typer.Option("experiment_two.json", help="Output path"),
    num_cases: int = typer.Option(10_000, help="Number of experiments"),
    delay_s: float = typer.Option(0.5, help="Delay between requests in seconds"),
) -> None:
    experiment_two(adverts_path, categories_path, target_path, num_cases, delay_s)


@app.command()
def three(
    adverts_path: str = typer.Option("/adverts.json", help="Path to the adverts json file"),
    categories_path: str = typer.Option("/categories.json", help="Path to the categories json file"),
    category_pairs_path: str = typer.Option("/category_pairs.json", help="Path to the category_pairs json file"),
    target_path: str = typer.Option("/experiment_three.json", help="Output path"),
    num_cases: int = typer.Option(10_000, help="Number of experiments"),
    delay_s: float = typer.Option(0.5, help="Delay between requests in seconds"),
) -> None:
    experiment_three(adverts_path, categories_path, category_pairs_path, target_path, num_cases, delay_s)
