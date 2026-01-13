from src.app.config.config import config
from src.app.services.report_service import ReportService


def prepare_report() -> None:
    exp_path = config.static_path / "results"
    exp_files = {
        "Exp1": exp_path / "experiment_one.json",
        "Exp2": exp_path / "experiment_two.json",
        "Exp3": exp_path / "experiment_three.json",
    }

    out_dir = config.static_path / "report"
    service = ReportService()
    report_path = service.run(
        exp_files=exp_files, out_dir=out_dir,
    )
    print("done:", report_path)
