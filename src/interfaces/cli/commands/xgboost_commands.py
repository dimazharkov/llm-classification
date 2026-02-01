import typer

from src.app.config.config import config
from src.app.services.xgboost_service import XGBoostService
from src.infra.storage.os_helper import load_from_disc, save_to_disc

app = typer.Typer()

@app.command()
def prep() -> None:
    # формируем обучающую выборку, как разницу между всем датасетом и мтестовой выборкой
    # inf = load_from_disc('xgboost/inf.json')
    # ids_to_remove = {d["advert_text"] for d in inf}
    #
    # source = load_from_disc('source/adverts.json') # весь датасет
    # train = [d for d in source if d.get("text") not in ids_to_remove]
    # save_to_disc(train, f"xgboost/train.json")

    # разделение фильтрованного датасета на обучающую и тестовую
    service = XGBoostService()
    service.prep(
        raw_path='xgboost/adverts.json',
        dest_path='xgboost'
    )

@app.command()
def train() -> None:
    service = XGBoostService()
    service.train(
        train_path='xgboost/train.json',
        artifacts_path=f"{config.static_path}/xgboost/artifacts"
    )

@app.command()
def predict() -> None:
    service = XGBoostService()
    service.predict(
        inf_path='xgboost/inf.json',
        artifacts_path=f"{config.static_path}/xgboost/artifacts"
    )

@app.command()
def enrich() -> None:
    categories = load_from_disc('categories.json')
    category_index = {c['id']: c['title'] for c in categories}

    svm_pred = load_from_disc('xgboost/inf_pred.json')
    for svm in svm_pred:
        svm["advert_category"] = category_index[svm["true_category_id"]]
        svm["predicted_category"] = category_index[svm["pred_category_id"]]

    save_to_disc(svm_pred, 'xgboost/inf_pred.json')
