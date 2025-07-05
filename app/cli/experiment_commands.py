import typer

from app.controllers.experiment_controller import ExperimentController
from app.infrastructure.llm_clients.gemini_client import GeminiClient

app = typer.Typer()

@app.command()
def one(
        adverts_path: str = typer.Option(..., help="Путь к json с объявлениями"),
        categories_path: str = typer.Option(..., help="Путь к json c категориями"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
        num_cases: int = typer.Argument(30, help="Количество экспериментов"),
        rate_limit: int = typer.Argument(1, help="Задержка между запросами")
    ):
    llm_client = GeminiClient()
    ExperimentController(llm_client).experiment_one(
        adverts_path, categories_path, target_path, num_cases, rate_limit
    )

@app.command()
def two(
        adverts_path: str = typer.Option(..., help="Путь к json с объявлениями"),
        categories_path: str = typer.Option(..., help="Путь к json c категориями"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
        num_cases: int = typer.Option(30, help="Количество экспериментов"),
        rate_limit: int = typer.Argument(1, help="Задержка между запросами")
    ):
    llm_client = GeminiClient()
    ExperimentController(llm_client).experiment_two(
        adverts_path, categories_path, target_path, num_cases, rate_limit
    )
