.PHONY: test lint embed clean help build-ci build-dev build-prod test-ci run-dev run-prod clean-docker stop logs

# Default target
help:
	@echo "InsightSpike-AI Commands"
	@echo "========================"
	@echo "Local Development:"
	@echo "  test          - Run tests with poetry"
	@echo "  lint          - Run linting with ruff"
	@echo "  embed         - Run embedding test"
	@echo "  clean         - Clean up test files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  build-ci      - Build CI/testing Docker image"
	@echo "  build-dev     - Build development Docker image"  
	@echo "  build-prod    - Build production Docker image"
	@echo "  build-all     - Build all Docker images"
	@echo "  test-ci       - Run tests in CI container"
	@echo "  run-dev       - Start development environment with Jupyter"
	@echo "  run-prod      - Start production environment"
	@echo "  stop          - Stop all containers"
	@echo "  clean-docker  - Clean up containers and images"

# Local development targets
test:
	poetry run pytest --maxfail=1 --disable-warnings -q --cov=src

lint:
	poetry run ruff src

embed:
	poetry run insightspike embed --path data/raw/test_sentences.txt

clean:
	rm -rf data/raw/test_sentences.txt data/memory* data/*.pt data/*.json

# Docker build targets
build-ci:
	@echo "Building CI Docker image..."
	docker build -f docker/Dockerfile.ci -t insightspike-ai:ci .

build-dev:
	@echo "Building development Docker image..."
	docker build -f docker/Dockerfile.dev -t insightspike-ai:dev .

build-prod:
	@echo "Building production Docker image..."
	docker build -f docker/Dockerfile.prod -t insightspike-ai:prod .

build-all: build-ci build-dev build-prod

# Docker run targets
run-dev:
	@echo "Starting development environment..."
	docker-compose up -d dev

run-prod:
	@echo "Starting production environment..."
	docker-compose up -d prod redis

# Docker test targets
test-ci:
	@echo "Running tests in CI container..."
	docker run --rm \
		-v $(PWD)/src:/app/src:ro \
		-v $(PWD)/tests:/app/tests:ro \
		-e INSIGHTSPIKE_LITE_MODE=1 \
		-e PYTHONPATH=/app/src \
		insightspike-ai:ci \
		python -m pytest tests/ -v --tb=short

# Docker utility targets
stop:
	@echo "Stopping all containers..."
	docker-compose down

clean-docker:
	@echo "Cleaning up containers and images..."
	docker-compose down --rmi local
	docker system prune -f
