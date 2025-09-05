import typer

from interfaces.cli.handlers.category_handlers import category_compare, category_enrich, category_preprocess

app = typer.Typer()

SELECTED_IDS = [
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
def preprocess(
    source_path: str = typer.Option("/source/categories.json", help="Путь для чтения"),
    target_path: str = typer.Option("raw_categories.json", help="Путь для сохранения"),
    selected_ids: list[int] = typer.Option(
        SELECTED_IDS,
        help="Список ID категорий, например: 11239 11243 595",
    ),
):
    category_preprocess(source_path, target_path, list(selected_ids))


@app.command()
def enrich(
    categories_path: str = typer.Option("raw_categories.json", help="Путь к json c категориями"),
    adverts_path: str = typer.Option("raw_adverts.json", help="Путь к json с объявлениями"),
    target_path: str = typer.Option("categories.json", help="Путь для сохранения"),
    top_k: int = typer.Option(20, help="Количество слов в BoW"),
) -> None:
    category_enrich(categories_path, adverts_path, target_path, top_k)


@app.command()
def pair_description(
    adverts_path: str = typer.Option("adverts.json", help="Путь к json с объявлениями"),
    categories_path: str = typer.Option("categories.json", help="Путь к json c категориями"),
    category_pairs_path: str = typer.Option("category_pairs.json", help="Путь для сохранения"),
) -> None:
    category_compare(adverts_path, categories_path, category_pairs_path)
