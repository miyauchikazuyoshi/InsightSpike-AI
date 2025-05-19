#!/usr/bin/env bash
# run inside Colab

poetry run pip install -q --upgrade pip
poetry run pip install -q torch==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
poetry run pip install -q torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
poetry run pip install -q torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
poetry run pip install -q torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
poetry run pip install -q torch-geometric==2.5.2

poetry run pip install -q -r requirements-colab.txt  # その他依存のみ

python -m insightspike.cli embed