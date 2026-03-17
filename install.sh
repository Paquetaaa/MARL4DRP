#!/bin/bash
set -e

pip install setuptools==65.5.0 wheel==0.38.4 "pip==23.3.1"
pip install gym==0.21.0 --no-build-isolation
pip install -r requirements.txt
pip install -e .
