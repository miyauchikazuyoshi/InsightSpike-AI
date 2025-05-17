#!/usr/bin/env bash
# run inside Colab

pip install -q -r requirements-colab.txt
python -m insightspike.cli embed
