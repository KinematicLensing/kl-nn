#!/bin/bash
# Submit a kl-nn slurm launcher with its log written to the configured log directory.
#
#   ./submit.sh arch/train_npe.slurm [sbatch options] [-- script arguments]
#
# The log goes to <logs>/<launcher stem>_%j.out (or _%A_%a.out for array jobs),
# where <logs> comes from paths.json (key "logs"). KLNN_REPO_ROOT is exported to
# the job, so launchers find the repo regardless of the submit directory.
# Any sbatch option is passed through, e.g. --array=0-9, --dependency=..., --test-only.
set -euo pipefail

if (( $# < 1 )); then
    sed -n '2,9p' "$0" >&2
    exit 2
fi

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "${KLNN_REPO_ROOT}/klnn_paths.sh"

script="$1"
shift
if [[ ! -f "${script}" ]]; then
    echo "submit.sh: no such launcher: ${script}" >&2
    exit 2
fi

sbatch_args=()
while (( $# > 0 )); do
    if [[ "$1" == "--" ]]; then
        shift
        break
    fi
    sbatch_args+=("$1")
    shift
done

stem="$(basename "${script}")"
stem="${stem%.*}"
pattern="%j"
if grep -qE '^#SBATCH[[:space:]]+(--array|-a)' "${script}" \
    || printf '%s\n' "${sbatch_args[@]+"${sbatch_args[@]}"}" | grep -qE '^(--array|-a)'; then
    pattern="%A_%a"
fi

mkdir -p "${KLNN_LOGS}"
exec sbatch --export=ALL,KLNN_REPO_ROOT="${KLNN_REPO_ROOT}" \
    --output="${KLNN_LOGS}/${stem}_${pattern}.out" \
    "${sbatch_args[@]+"${sbatch_args[@]}"}" "${script}" "$@"
