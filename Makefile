.PHONY: test lint embed clean help

# Default target
help:
	@echo "InsightSpike-AI Commands"
	@echo "========================"
	@echo "  test          - Run tests with poetry"
	@echo "  lint          - Run linting with ruff"
	@echo "  embed         - Run embedding test"
	@echo "  clean         - Clean up test files"

# Local development targets
test:
	poetry run pytest --maxfail=1 --disable-warnings -q --cov=src

lint:
	poetry run ruff src

embed:
	poetry run insightspike embed --path data/raw/test_sentences.txt

clean:
	rm -rf data/raw/test_sentences.txt data/memory* data/*.pt data/*.json
