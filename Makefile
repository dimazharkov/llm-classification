VENV := ./venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
FLAKE8 := $(VENV)/bin/flake8

TARGETS := src

.PHONY: venv install fmt fmt-check type-check complexity lint

venv:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) -q install --upgrade pip

install: venv
	$(PIP) install black ruff mypy flake8

fmt:
	$(RUFF) check --fix $(TARGETS)
	$(BLACK) $(TARGETS)

fmt-check:
	$(BLACK) --check $(TARGETS)
	$(RUFF) check $(TARGETS)

type-check:
	$(MYPY) $(TARGETS)

complexity:
	$(FLAKE8) $(TARGETS)

lint: fmt-check type-check

experiment-one:
	python -m src.interfaces.cli.main experiment one

experiment-two:
	python -m src.interfaces.cli.main experiment two

experiment-three:
	python -m src.interfaces.cli.main experiment three
