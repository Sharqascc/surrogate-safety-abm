.PHONY: bootstrap test audit clean
bootstrap:
	@python scripts/dev.py bootstrap
test:
	@python scripts/dev.py test
audit:
	@python scripts/dev.py audit
clean:
	@rm -rf .pytest_cache .venv-bootstrapped
	@find . -name __pycache__ -prune -exec rm -rf {} +
