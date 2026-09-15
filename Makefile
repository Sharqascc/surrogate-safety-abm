.PHONY: install check ruff mypy test test-unit test-property test-integration clean

install:
	pip install -e ".[dev]"

check: ruff mypy test
	@echo "All quality checks passed."

ruff:
	ruff check src tests
	ruff format --check src tests

mypy:
	mypy src

test:
	pytest -q

test-unit:
	pytest tests/unit -v

test-property:
	pytest tests/property -v

test-integration:
	pytest tests/integration -v -m "not slow"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml .hypothesis
	find . -type d -name __pycache__ -exec rm -rf {} +
