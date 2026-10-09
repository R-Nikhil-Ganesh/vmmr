# ==============================================================================
# Single source of truth for machine-specific paths (sourced by every .sh runner,
# and read by paths.py for the Python scripts).
#
# Moving to a new server:
#   1. Copy nothing. Create  paths.local.sh  next to this file (it is gitignored) with only
#      the values that differ, e.g.:
#          ECON_ROOT=/data/Econ
#          PYTHON_BIN=/opt/envs/pt-env/bin/python
#   2. Check the result:   python paths.py --check
#
# Every variable can also be overridden for a single run from the shell:
#          ECON_ROOT=/data/Econ bash run_all_pipeline.sh
# Defaults below are the original server's values, so behavior is unchanged until you override them.
# ==============================================================================

_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Per-machine overrides (gitignored). Loaded first so ':=' defaults below do not clobber them.
# Precedence: shell environment > paths.local.sh > defaults below.
if [ -f "${_PATHS_DIR}/paths.local.sh" ]; then
    _SAVED=""
    for _k in ECON_ROOT PM_IMG_DIR PM_MANIFEST_CSV PM_LABEL_MAP EXT_DATASETS_DIR PYTHON_BIN AUTO_GIT_PUSH REPO_ROOT; do
        if [ -n "${!_k+x}" ]; then _SAVED="${_SAVED} ${_k}=$(printf '%q' "${!_k}")"; fi
    done
    # shellcheck disable=SC1091
    . "${_PATHS_DIR}/paths.local.sh"
    eval "${_SAVED}"   # re-apply anything that was already set in the environment
    unset _SAVED _k
fi

# Repo layout (derived from this file's location, no need to set)
: "${MOBILENET_DIR:=${_PATHS_DIR}}"
: "${REPO_ROOT:=$(cd "${_PATHS_DIR}/../../.." && pwd)}"

# Data root: everything below is derived from it unless set explicitly
: "${ECON_ROOT:=/home/researchadmin/Econ}"

# PlatesMania
: "${PM_IMG_DIR:=${ECON_ROOT}/resized_640x640}"
: "${PM_MANIFEST_CSV:=${ECON_ROOT}/models/dataset_manifests/dataset_1235models_splits.csv}"
: "${PM_LABEL_MAP:=${ECON_ROOT}/models/dataset_manifests/label_map_1235models.json}"

# External datasets (raw sources used by external_dataset/prepare_external_1235models.py)
: "${EXT_DATASETS_DIR:=${ECON_ROOT}/external_datasets}"

# Python interpreter: <repo parent>/pt-env if present, else python3
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$(dirname "${REPO_ROOT}")/pt-env/bin/python" ]; then
        PYTHON_BIN="$(dirname "${REPO_ROOT}")/pt-env/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi

# Commit and push results when a runner finishes (1 = on, 0 = off)
: "${AUTO_GIT_PUSH:=1}"

export MOBILENET_DIR REPO_ROOT ECON_ROOT PM_IMG_DIR PM_MANIFEST_CSV PM_LABEL_MAP EXT_DATASETS_DIR PYTHON_BIN AUTO_GIT_PUSH
