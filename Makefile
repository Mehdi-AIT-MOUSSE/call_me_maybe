.PHONY: install run debug clean lint lint-strict

run:
	uv run python3 -m src ${ARGS}

install:
	uv add flake8 
	uv add mypy
	uv add numpy 
	uv add pydantic
	uv add ./llm_sdk
	uv sync


debug:
	uv run python3 -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

