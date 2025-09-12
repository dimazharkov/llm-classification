VENV := ./venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
FLAKE8 := $(VENV)/bin/flake8
TARGETS := src

IMAGE_NAME := myapp
TAG        := latest
CONTAINER  := myapp

.PHONY: venv install fmt fmt-check type-check complexity lint exp-one exp-two exp-three build create start start rmi clean logs shell run-one run-two run-three

# local

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

exp-one:
	python -m src.interfaces.cli.main experiment one

exp-two:
	python -m src.interfaces.cli.main experiment two

exp-three:
	python -m src.interfaces.cli.main experiment three

# docker

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

create: build
	docker run -d --name $(CONTAINER) --env-file .env $(IMAGE_NAME):$(TAG)

start: create
	docker start $(CONTAINER)

stop:
	docker stop $(CONTAINER)

rmi:
	- docker rmi $(IMAGE_NAME):$(TAG)

clean: stop rmi

logs:
	docker logs -f $(CONTAINER)

shell:
	docker exec -it $(CONTAINER) bash

run-one:
	docker exec -it $(CONTAINER) python -m src.main experiment one

run-two:
	docker exec -it $(CONTAINER) python -m src.main experiment two

run-three:
	docker exec -it $(CONTAINER) python -m src.main experiment three