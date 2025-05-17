FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# install Python deps
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

# application source
COPY . .
RUN pip install git+https://github.com/timothybrooks/fsq.git@v0.1.0#egg=fsq-pytorch \
        networkx torch-geometric gmatch4py faiss-gpu ragatouille

CMD ["python", "-m", "insightspike.cli"]

