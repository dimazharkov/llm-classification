import typer

from interfaces.cli.commands import experiment_commands

app = typer.Typer()
# app.add_typer(category_commands.app, name="category")
# app.add_typer(advert_commands.app, name="advert")
app.add_typer(experiment_commands.app, name="experiment")
# app.add_typer(debug_commands.app, name="test")

if __name__ == "__main__":
    app()
