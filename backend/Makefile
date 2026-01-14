PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: venv install dev test run clean

venv:
	$(PYTHON) -m venv --clear $(VENV)
	$(BIN)/python -m pip install -U pip
	
install: venv
	$(BIN)/pip install .

dev: venv
	$(BIN)/pip install -e .[dev]

test: dev
	$(BIN)/pytest -q

run:
	. .venv/bin/activate && uvicorn app.main:app --reload --port 8000

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist build *.egg-info