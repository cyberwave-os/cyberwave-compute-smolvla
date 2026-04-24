#!/usr/bin/env bash
# Install LeRobot with SmolVLA dependencies for `smolva-so101.py` and SmolVLA policies.
# Requires Python >= 3.12 (see pyproject.toml). Prefer a virtual environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: $PY not found. Set PYTHON=/path/to/python3 or install Python 3.12+." >&2
  exit 1
fi

# Must match cyberwave.yml (source "$HOME/.venv/smolvla/bin/activate" && python …).
# Override with SMOLVLA_VENV if needed. Using this venv avoids PEP 668 errors on
# Homebrew / other "externally managed" system Pythons.
VENV_DIR="${SMOLVLA_VENV:-$HOME/.venv/smolvla}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PY" -m venv "$VENV_DIR"
  else
    echo "Using existing virtual environment at $VENV_DIR"
  fi
  PY="${VENV_DIR}/bin/python"
  if [[ ! -x "$PY" ]]; then
    echo "error: expected venv python at $PY" >&2
    exit 1
  fi
fi

# lerobot + draccus policy config parsing breaks on Python 3.14+ (argparse / union types).
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3, 14) else 1)'; then
  echo "error: SmolVLA training requires Python 3.12 or 3.13 (not 3.14+). Recreate the venv:" >&2
  echo "  rm -rf \"${SMOLVLA_VENV:-$HOME/.venv/smolvla}\" && PYTHON=python3.12 ./install.sh" >&2
  exit 1
fi

echo "Installing dependencies from requirements.txt"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"

# Verify cyberwave import works
if ! "$PY" -c "from cyberwave import Cyberwave" 2>/dev/null; then
  echo "error: cyberwave import failed after install" >&2
  exit 1
fi
echo "✓ cyberwave SDK installed and verified"

echo
echo "Done. Example:"
echo "  export SMOLVLA_CHECKPOINT=your-org/your-model   # optional: pass --checkpoint instead"
echo "  $PY smolva-so101.py --prompt '...' --state-file state.npy --images-json images.json"
