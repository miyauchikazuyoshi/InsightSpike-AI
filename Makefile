.PHONY: test lint embed clean

test:
    poetry run pytest --maxfail=1 --disable-warnings -q --cov=src

lint:
    poetry run ruff src

embed:
    poetry run insightspike embed --path data/raw/test_sentences.txt

clean:
    rm -rf data/raw/test_sentences.txt data/memory* data/*.pt data/*.json