import typer

from src.app.config.config import config
from src.app.services.svm_service import SVMService
from src.infra.storage.file_helper import read_json, write_json
from src.infra.storage.os_helper import load_from_disc, save_to_disc

app = typer.Typer()

@app.command()
def train() -> None:
    service = SVMService()
    service.train(
        train_path='svm/train.json',
        artifacts_path=f"{config.static_path}/svm/artifacts"
    )

@app.command()
def predict() -> None:
    service = SVMService()
    service.predict(
        inf_path='svm/inf.json',
        artifacts_path=f"{config.static_path}/svm/artifacts"
    )

@app.command()
def enrich() -> None:
    categories = load_from_disc('categories.json')
    category_index = {c['id']: c['title'] for c in categories}

    svm_pred = load_from_disc('svm/inf_pred.json')
    updated = []
    for svm in svm_pred:
        svm["advert_category"] = category_index[svm["true_category_id"]]
        svm["predicted_category"] = category_index[svm["pred_category_id"]]

    save_to_disc(svm_pred, 'svm/inf_pred.json')