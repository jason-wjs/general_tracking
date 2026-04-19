.PHONY: test lint typecheck

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group dev pytest

lint:
	uv run --group dev ruff check .

typecheck:
	uv run --group dev pyright
