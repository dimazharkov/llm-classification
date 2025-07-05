import typer

from app.cli import category_commands, advert_commands, experiment_commands

app = typer.Typer()
app.add_typer(category_commands.app, name="category")
app.add_typer(advert_commands.app, name="advert")
app.add_typer(experiment_commands.app, name="experiment")

if __name__ == "__main__":
    app()
