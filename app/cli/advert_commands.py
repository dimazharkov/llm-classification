import typer

from app.controllers.advert_controller import AdvertController
from app.infrastructure.llm_clients.gemini_client import GeminiClient

app = typer.Typer()

DEFAULT_SELECTED_IDS = [
    11239,
    11243,
    595,
    596,
    597,
    599,
    11251,
    722,
    723,
    729,
    822,
    823,
    826,
    827,
    1073,
]


@app.command()
def summarize(
    source_path: str = typer.Option("raw_categories.json", help="Путь для чтения"),
    target_path: str = typer.Option("categories.json", help="Путь для сохранения"),
):
    llm_client = GeminiClient()
    AdvertController().summarize(source_path, target_path, llm_client)


@app.command()
def preprocess(
    source_path: str = typer.Option("/source/adverts.json", help="Путь для чтения"),
    target_path: str = typer.Option("raw_adverts.json", help="Путь для сохранения"),
    per_category: int = typer.Option(30, help="Объявлений на категорию"),
    selected_ids: list[int] = typer.Option(DEFAULT_SELECTED_IDS, help="Список ID категорий, например: 11239 11243 595"),
):
    AdvertController().preprocess(source_path, target_path, per_category, list(selected_ids))

@app.command()
def indexing(
    source_path: str = typer.Option("adverts.json", help="Путь для чтения"),
    target_path: str = typer.Option("indexed_adverts.json", help="Путь для сохранения"),
):
    AdvertController().indexing(source_path, target_path)

@app.command()
def analyze(
    source_path: str = typer.Option("experiment_one.json", help="Путь для чтения"),
    rejected_path: str = typer.Option("rejected_ads.json", help="Путь для сохранения"),
):
    AdvertController().analyze_results(
        source_path,
        rejected_path,
    )
