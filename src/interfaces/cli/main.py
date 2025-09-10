import typer

from src.interfaces.cli.commands import category_commands, experiment_commands

app = typer.Typer()
app.add_typer(experiment_commands.app, name="experiment")
app.add_typer(category_commands.app, name="category")
