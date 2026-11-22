.PHONY: bootstrap test audit clean
bootstrap:
	@python scripts/main.py bootstrap
test:
	@python scripts/main.py test
audit:
	@python scripts/main.py audit
clean:
	@rm -rf .pytest_cache .venv-bootstrapped
	@find . -name __pycache__ -prune -exec rm -rf {} +
