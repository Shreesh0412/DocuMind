#!/usr/bin/env bash
set -euo pipefail

# Run inside the project's activated virtual environment.
python -m compileall -q app.py src scripts tests
python -m unittest discover -s tests -v
python -m pip check
python -m pip freeze > requirements.lock.txt
printf '\nCreated requirements.lock.txt from the tested active environment.\n'
