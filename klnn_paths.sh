#!/bin/bash
# Shell access to the kl-nn path config (paths.json). Source this file:
#
#   source "${KLNN_REPO_ROOT}/klnn_paths.sh"
#   klnn_path models          # -> /ocean/.../models
#   klnn_activate             # module load anaconda3 + conda activate <conda_env>
#
# Sourcing also exports KLNN_REPO_ROOT and one KLNN_<KEY> variable per config key
# (KLNN_DATASETS, KLNN_MODELS, KLNN_LOGS, KLNN_CONDA_ENV, ...). Launchers use them as
# defaults, e.g. MODEL_ROOT="${MODEL_ROOT:-${KLNN_MODELS}}", so overrides keep working.

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export KLNN_REPO_ROOT

# The resolver is stdlib-only and Python-3.6 compatible, so it runs before any env is active.
if [[ -x /usr/bin/python3 ]]; then
    KLNN_PATHS_PYTHON="${KLNN_PATHS_PYTHON:-/usr/bin/python3}"
else
    KLNN_PATHS_PYTHON="${KLNN_PATHS_PYTHON:-python3}"
fi

klnn_path() {
    "${KLNN_PATHS_PYTHON}" "${KLNN_REPO_ROOT}/klnn_paths.py" "$1"
}

klnn_activate() {
    module load anaconda3 >/dev/null 2>&1 || true
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${KLNN_CONDA_ENV}"
}

_klnn_all="$("${KLNN_PATHS_PYTHON}" "${KLNN_REPO_ROOT}/klnn_paths.py" --all)" || {
    echo "klnn_paths.sh: failed to read ${KLNN_REPO_ROOT}/paths.json" >&2
    return 1 2>/dev/null || exit 1
}
# Every key in paths.json is exported as KLNN_<KEY> (e.g. KLNN_MODELS, KLNN_DATASETS),
# unless already set in the environment.
while IFS='=' read -r _klnn_key _klnn_val; do
    [[ -z "${_klnn_key}" ]] && continue
    _klnn_var="KLNN_${_klnn_key^^}"
    if [[ -z "${!_klnn_var:-}" ]]; then
        export "${_klnn_var}=${_klnn_val}"
    fi
done <<< "${_klnn_all}"
unset _klnn_key _klnn_val _klnn_var
unset _klnn_all
