# Multi-stage build for InsightSpike-AI
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Copy source code (needed for Poetry to install the package)
COPY src/ ./src/

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM base as production

# Copy additional data and experiments
COPY data/ ./data/
COPY experiments/ ./experiments/

# Install PyTorch CPU version for production
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install FAISS CPU version
RUN pip install faiss-cpu

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV INSIGHTSPIKE_MODE=production

# Create non-root user
RUN useradd -m -u 1000 insightspike && chown -R insightspike:insightspike /app
USER insightspike

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import insightspike; print('OK')" || exit 1

# Default command
CMD ["poetry", "run", "python", "-m", "insightspike", "--help"]

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install && rm -rf $POETRY_CACHE_DIR

# Install development tools
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu

# Copy everything for development
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV INSIGHTSPIKE_MODE=development

# Default command for development
CMD ["poetry", "shell"]

# Colab stage
FROM base as colab

# Install Colab-specific dependencies
RUN poetry install --with colab --with docker && rm -rf $POETRY_CACHE_DIR

# Install PyTorch CPU for Colab
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu

# Copy source and notebooks
COPY . .

# Colab-style environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV INSIGHTSPIKE_LITE_MODE=1
ENV JUPYTER_ENABLE_LAB=yes

# Create Colab-compatible user
RUN useradd -m -u 1000 colab && chown -R colab:colab /app
USER colab

# Expose Jupyter port
EXPOSE 8888

# Default command for Colab
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
