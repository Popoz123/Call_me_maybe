.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	rm -rf .mypy_cache src/__pycache__

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs src

lint-strict:
	flake8 src
	mypy --strict src