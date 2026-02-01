import typer

from src.app.config.config import config
from src.app.services.rf_service import RFService
from src.app.services.svm_service import SVMService
from src.infra.storage.os_helper import load_from_disc, save_to_disc

app = typer.Typer()

@app.command()
def train() -> None:
    service = RFService()
    service.train(
        train_path='rf/train.json',
        artifacts_path=f"{config.static_path}/rf/artifacts"
    )

@app.command()
def predict() -> None:
    service = RFService()
    service.predict(
        inf_path='rf/inf.json',
        artifacts_path=f"{config.static_path}/rf/artifacts"
    )

@app.command()
def enrich() -> None:
    categories = load_from_disc('categories.json')
    category_index = {c['id']: c['title'] for c in categories}

    svm_pred = load_from_disc('rf/inf_pred.json')
    for svm in svm_pred:
        svm["advert_category"] = category_index[svm["true_category_id"]]
        svm["predicted_category"] = category_index[svm["pred_category_id"]]

    save_to_disc(svm_pred, 'rf/inf_pred.json')