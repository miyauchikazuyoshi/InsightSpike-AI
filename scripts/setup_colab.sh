#!/usr/bin/env bash
# run inside Colab

curl -sSL https://raw.githubusercontent.com/<USER>/<REPO>/main/requirements-colab.txt -o /tmp/req.txt
pip install -q -r /tmp/req.txt
python -m insightspike.cli embed
