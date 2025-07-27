import typer

from app.core.domain.category import Category
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.infrastructure.di import Container
from app.repositories.category_pair_diff_repository import CategoryPairDiffRepository

app = typer.Typer()


def init_container():
    container = Container()
    container.config.ADVERTS_FILE_PATH.from_value("adverts.json")
    container.config.CATEGORIES_FILE_PATH.from_value("categories.json")
    container.config.TARGET_FILE_PATH.from_value("test.json")
    container.config.CATEGORIES_PAIR_FILE_PATH.from_value("categories_pair.json")
    return container


@app.command()
def ex_three():
    container = init_container()
    predict_five_categories_use_case = container.predict_five_categories_use_case()

    adverts_repo = container.advert_file_repo()
    advert = adverts_repo.get()[0]

    category_list: list[Category] = predict_five_categories_use_case.run(advert)
    # print("category_list -----")
    # print(category_list)

    compare_category_pair_use_case = container.compare_category_pair_use_case()
    category_pair_diff_repo: CategoryPairDiffRepository = compare_category_pair_use_case.run(category_list, rate_limit=1)
    print("category_pair_diff_repo -----")
    print(category_pair_diff_repo.all())

    pairwise_classification_use_case = container.pairwise_classification_use_case()
    advert_category_prediction: AdvertCategoryPrediction | None = pairwise_classification_use_case.run(
        advert,
        category_pair_diff_repo,
        rate_limit=1,
    )

    print("advert_category_prediction -----")
    print(advert_category_prediction)
