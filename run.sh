#!/usr/bin/env bash
set -e
python3 -m venv .venv
.venv/bin/pip install -q .
MLSWEEP_SITE=$(ls -d /tmp/mlsweep_venv/lib/python*/site-packages 2>/dev/null | head -1)
export PYTHONPATH="${MLSWEEP_SITE}${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python "$@"
