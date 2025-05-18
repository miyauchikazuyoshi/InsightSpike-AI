FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# install Python deps
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction

# (Optional) Upgrade torch to 2.2.2 if strict version match is needed
RUN pip install --upgrade torch==2.2.2

# application source
COPY . .
RUN pip install git+https://github.com/timothybrooks/fsq.git@v0.1.0#egg=fsq-pytorch \
        networkx torch-geometric gmatch4py faiss-gpu-cu11 ragatouille

CMD ["python", "-m", "insightspike.cli"]

