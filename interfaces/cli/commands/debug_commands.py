import typer
from interfaces.cli.handlers.debug_handlers import debug_experiment

app = typer.Typer()

@app.command()
def ex_three():
    debug_experiment()