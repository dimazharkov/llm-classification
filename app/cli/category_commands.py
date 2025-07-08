from typing import List, Tuple
import typer
from app.controllers.category_controller import CategoryController

app = typer.Typer()

@app.command()
def preprocess(
        source_path: str = typer.Option("/source/categories.json", help="Путь для чтения"),
        target_path: str = typer.Option("raw_categories.json", help="Путь для сохранения"),
        selected_ids: List[int] = typer.Option([11239, 11243, 595, 596, 597, 599, 11251, 722, 723, 729, 822, 823, 826, 827, 1073], help="Список ID категорий, например: 11239 11243 595")
        # selected_ids: List[int] = typer.Option("11239 11243 595 596 597 599 11251 722 723 729 822 823 826 827 1073", help="ID категорий, например: 11239 11243 595")
    ):
    CategoryController().preprocess(
        source_path, target_path, list(selected_ids)
    )

@app.command()
def enrich(
        categories_path: str = typer.Option("raw_categories.json", help="Путь к json c категориями"),
        adverts_path: str = typer.Option("raw_adverts.json", help="Путь к json с объявлениями"),
        target_path: str = typer.Option("categories.json", help="Путь для сохранения"),
        top_k: int = typer.Option(20, help="Количество слов в BoW")
    ):
    CategoryController().build_bow(
        categories_path, adverts_path, target_path, top_k
    )
