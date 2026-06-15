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

echo "Upgrading pip"
"$PY" -m pip install --upgrade pip setuptools wheel

# Install PyTorch before lerobot[smolvla] so CPU-only Docker / macOS builds do not
# pull a CUDA wheel by default.
if ! "$PY" -c "import torch" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "Installing PyTorch (CUDA 12.1)"
    "$PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    echo "Installing PyTorch (CPU-only)"
    "$PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
else
  echo "PyTorch already installed: $("$PY" -c 'import torch; print(torch.__version__)')"
fi

echo "Installing dependencies from requirements.txt"
"$PY" -m pip install -r "$ROOT/requirements.txt"

# Verify cyberwave import works
if ! "$PY" -c "from cyberwave import Cyberwave" 2>/dev/null; then
  echo "error: cyberwave import failed after install" >&2
  exit 1
fi
echo "✓ cyberwave SDK installed and verified"

# ── Pre-fetch model weights ──────────────────────────────────────────────────
#
# Defaults to SMOLVLA_DEFAULT_CHECKPOINT (set in the Dockerfile ENV).
# Override with SMOLVLA_PREFETCH_CHECKPOINT=<repo>, or set it to "" to skip.
_PREFETCH="${SMOLVLA_PREFETCH_CHECKPOINT-${SMOLVLA_DEFAULT_CHECKPOINT:-lerobot/smolvla_base}}"
if [[ -n "${_PREFETCH}" ]]; then
  _HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
  # HF names the cache dir as models--{org}--{name} (the / separator becomes --)
  _CACHE_KEY="models--${_PREFETCH/\//"--"}"
  _SNAPSHOTS="${_HF_CACHE}/hub/${_CACHE_KEY}/snapshots"
  if [[ -d "${_SNAPSHOTS}" ]] && [[ -n "$(ls -A "${_SNAPSHOTS}" 2>/dev/null)" ]]; then
    echo "Model weights already cached: ${_PREFETCH} (${_SNAPSHOTS})"
  else
    echo "Pre-fetching model weights: ${_PREFETCH}"
    PREFETCH_REPO="${_PREFETCH}" "$PY" - <<'PYEOF'
import os, sys
repo = os.environ["PREFETCH_REPO"]
if repo.startswith("/") or repo.startswith("."):
    print(f"Skipping prefetch: local path {repo}")
    sys.exit(0)
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo)
    print(f"✓ weights cached: {repo}")
except Exception as exc:
    print(f"warning: could not pre-fetch {repo}: {exc}", file=sys.stderr)
PYEOF
  fi
fi

echo
echo "Done. Example:"
echo "  export SMOLVLA_CHECKPOINT=your-org/your-model   # optional: pass --checkpoint instead"
echo "  $PY smolva-so101.py --prompt '...' --state-file state.npy --images-json images.json"
