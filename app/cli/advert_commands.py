from typing import List

import typer

from app.controllers.advert_controller import AdvertController
from app.infrastructure.llm_clients.gemini_client import GeminiClient

app = typer.Typer()

@app.command()
def summarize(
        source_path: str = typer.Option(..., help="Путь для чтения"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
):
    llm_client = GeminiClient()
    AdvertController().summarize(
        source_path, target_path, llm_client
    )

@app.command()
def preprocess(
        source_path: str = typer.Option(..., help="Путь для чтения"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
        per_category: int = typer.Option(..., help="Объявлений на категорию"),
        selected_ids: List[int] = typer.Argument(None, help="ID категорий, например: 11239 11243 595")

    ):
    AdvertController().preprocess(
        source_path, target_path, per_category, list(selected_ids)
    )

