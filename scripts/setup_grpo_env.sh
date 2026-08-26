#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

readonly ENV_DIR="${GENPROVE_ENV_DIR:-${REPO_ROOT}/.venv}"
readonly PYTHON_BIN="${ENV_DIR}/bin/python"
readonly REQUIREMENTS_FILE="${GENPROVE_REQUIREMENTS_FILE:-${REPO_ROOT}/requirements-training.txt}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    printf 'Missing requirements file: %s\n' "${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'python3 is required to create %s\n' "${ENV_DIR}" >&2
        exit 1
    fi
    python3 -m venv "${ENV_DIR}" || {
        printf 'Could not create %s. Install the system python3-venv package first.\n' "${ENV_DIR}" >&2
        exit 1
    }
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --requirement "${REQUIREMENTS_FILE}"

"${PYTHON_BIN}" - <<'PY'
import accelerate
import deepspeed
import datasets
import modelscope
import sentence_transformers
import torch
import transformers
import trl
import swift

print("Environment check passed")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("ms-swift", getattr(swift, "__version__", "unknown"))
print("sentence-transformers", sentence_transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("modelscope", modelscope.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
PY
