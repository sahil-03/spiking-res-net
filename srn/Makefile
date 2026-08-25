.PHONY: lint-fix check

# Run from package dir; tools use the repo root (parent).
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

lint-fix:
	cd $(ROOT) && uv run ruff check --fix .
	cd $(ROOT) && uv run ruff format .

check: lint-fix
	cd $(ROOT) && uv run mypy
