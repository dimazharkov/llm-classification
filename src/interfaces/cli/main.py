import typer

from src.interfaces.cli.commands import category_commands, experiment_commands, report_commands, xgboost_commands, \
    svm_commands, rf_commands

app = typer.Typer()
app.add_typer(experiment_commands.app, name="experiment")
app.add_typer(category_commands.app, name="category")
app.add_typer(report_commands.app, name="report")
app.add_typer(xgboost_commands.app, name="xgboost")
app.add_typer(svm_commands.app, name="svm")
app.add_typer(rf_commands.app, name="rf")
