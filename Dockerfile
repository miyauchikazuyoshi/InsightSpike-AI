FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# OS パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
      libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 環境
WORKDIR /workspace
COPY pyproject.toml poetry.lock ./
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction

# ソースを追加
COPY . .

CMD ["python", "-m", "insightspike.cli"]

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# …前段で apt や poetry を入れるステップ…

RUN pip install \
      git+https://github.com/timothybrooks/fsq.git@v0.1.0#egg=fsq-pytorch \
      networkx torch-geometric gmatch4py faiss-gpu ragatouille
