from typing import List, Tuple
import typer
from app.controllers.category_controller import CategoryController

app = typer.Typer()

@app.command()
def preprocess(
        source_path: str = typer.Option(..., help="Путь для чтения"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
        selected_ids: List[int] = typer.Argument(None, help="ID категорий, например: 11239 11243 595")
    ):
    CategoryController().preprocess(
        source_path, target_path, list(selected_ids)
    )

@app.command()
def build_bow(
        categories_path: str = typer.Option(..., help="Путь к json c категориями"),
        adverts_path: str = typer.Option(..., help="Путь к json с объявлениями"),
        target_path: str = typer.Option(..., help="Путь для сохранения"),
        top_k: int = typer.Argument(None, help="Количество слов в BoW")
    ):
    CategoryController().build_bow(
        categories_path, adverts_path, target_path, top_k
    )
