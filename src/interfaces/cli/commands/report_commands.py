import typer

from src.interfaces.cli.handlers.report_handlers import prepare_report

app = typer.Typer()

@app.command()
def prepare() -> None:
    prepare_report()