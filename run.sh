#!/usr/bin/env bash
# Run any CLI in this directory with PYTHONPATH set to the package root.
#
# Usage:
#   ./run.sh stage_a.py freeze --config config.yaml --dataset beemachine
#   ./run.sh stage_b.py baseline --backbone convnext_nano.in12k
#   PYTHON=python3.11 ./run.sh stage_c.py ddp --mode whole
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: '${PYTHON}' not found. Set PYTHON to a Python 3 interpreter with the requirements installed." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <script.py | -c code | -m module ...> [args...]" >&2
  exit 2
fi

exec "${PYTHON}" "$@"
