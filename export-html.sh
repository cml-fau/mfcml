#!/usr/bin/env bash
# Export notebooks to HTML, honouring the cell tags in jupyter_nbconvert_config.py.
# Usage: ./export-html.sh [notebook ...]   (no arguments = all notebooks)
set -euo pipefail
cd "$(dirname "$0")"

notebooks=("$@")
if [ ${#notebooks[@]} -eq 0 ]; then
    notebooks=(winter-2026/*.ipynb)
fi

exec .venv/bin/jupyter nbconvert \
    --config jupyter_nbconvert_config.py \
    --to html "${notebooks[@]}"
