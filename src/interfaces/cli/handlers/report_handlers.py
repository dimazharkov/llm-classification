from src.app.config.config import config
from src.app.services.report_service import ReportService


def prepare_report() -> None:
    exp_path = config.static_path / "2026-01-24"
    exp_files = {
        "XGB": exp_path / "xgboost" / "inf_pred.json",
        "SVM": exp_path / "svm" / "inf_pred.json",
        "RF": exp_path / "rf" / "inf_pred.json",
        "ZS": exp_path / "llm" / "experiment_one.json",
        "BOW": exp_path / "llm" / "experiment_two_bow.json",
        "TF-IDF": exp_path / "llm" / "experiment_two_idf.json",
        "PW 3": exp_path / "llm" / "experiment_three_3.json",
        "PW 5": exp_path / "llm" / "experiment_three_5.json",
    }

    out_dir = config.static_path / "2026-01-24" / "report"
    service = ReportService()
    report_path = service.run(
        exp_files=exp_files, out_dir=out_dir,
    )
    print("done:", report_path)
