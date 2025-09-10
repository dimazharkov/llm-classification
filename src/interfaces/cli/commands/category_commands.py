import typer

from src.interfaces.cli.handlers.category_handlers import category_prep_bow

app = typer.Typer()


@app.command()
def prep_bow(
    categories_path: str = typer.Option("raw_categories.json", help="Путь к json c категориями"),
    adverts_path: str = typer.Option("raw_adverts.json", help="Путь к json с объявлениями"),
    target_path: str = typer.Option("categories.json", help="Путь для сохранения"),
    top_k: int = typer.Option(20, help="Количество слов в BoW"),
) -> None:
    category_prep_bow(categories_path, adverts_path, target_path, top_k)
