#!/bin/bash
# Shell access to the kl-nn path config (paths.json). Source this file:
#
#   source "${KLNN_REPO_ROOT}/klnn_paths.sh"
#   klnn_path models          # -> /ocean/.../models
#   klnn_activate             # module load anaconda3 + conda activate <conda_env>
#   klnn_sbatch [opts] x.slurm # sbatch with the log path from paths.json
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

# Drop-in replacement for sbatch: same arguments (options, launcher, launcher args).
# Adds --output=<logs>/<launcher stem>_%j.out (or _%A_%a.out for array jobs) unless
# --output/-o is given, and passes KLNN_REPO_ROOT and KLNN_ANALYSIS_ROOT
# to the job, also when an explicit --export=VAR,... list is used.
klnn_sbatch() {
    local args=() script="" array=0 has_output=0 has_export=0 arg
    local roots="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},KLNN_ANALYSIS_ROOT=${KLNN_ANALYSIS_ROOT}"
    while (( $# > 0 )); do
        arg="$1"
        shift
        if [[ -z "${script}" && "${arg}" != -* ]]; then
            script="${arg}"
            break
        fi
        case "${arg}" in
            --output=*|-o) has_output=1 ;;
            --array=*|-a) array=1 ;;
            --export=NONE) ;;
            --export=*) has_export=1; arg="${arg},${roots}" ;;
        esac
        args+=("${arg}")
        # options that take their value as the next word
        if [[ "${arg}" =~ ^-[a-zA-Z]$ && $# -gt 0 ]]; then
            args+=("$1")
            shift
        fi
    done
    if [[ -z "${script}" || ! -f "${script}" ]]; then
        echo "klnn_sbatch: no launcher file in arguments (got '${script}')" >&2
        return 2
    fi
    if grep -qE '^#SBATCH[[:space:]]+(--array|-a)' "${script}"; then
        array=1
    fi
    if (( ! has_export )); then
        args+=("--export=ALL,${roots}")
    fi
    if (( ! has_output )); then
        local stem
        stem="$(basename "${script}")"
        stem="${stem%.*}"
        mkdir -p "${KLNN_LOGS}"
        if (( array )); then
            args+=("--output=${KLNN_LOGS}/${stem}_%A_%a.out")
        else
            args+=("--output=${KLNN_LOGS}/${stem}_%j.out")
        fi
    fi
    command sbatch "${args[@]}" "${script}" "$@"
}
