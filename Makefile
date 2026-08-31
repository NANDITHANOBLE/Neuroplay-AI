.PHONY: install install-dev format lint test run clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

format:
	black src tests main.py
	isort src tests main.py

lint:
	ruff check src tests main.py
	mypy src

test:
	pytest

run:
	python main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
powershell
notepad .github\workflows\ci.yml

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint (ruff)
        run: ruff check src tests main.py

      - name: Format check (black)
        run: black --check src tests main.py

      - name: Type check (mypy)
        run: mypy src

      - name: Run tests
        run: pytest
