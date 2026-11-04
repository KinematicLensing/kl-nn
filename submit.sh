#!/bin/bash
# Submit a kl-nn slurm launcher with its log written to the configured log directory.
#
#   ./submit.sh [sbatch options] arch/train_npe.slurm [launcher arguments]
#
# Same arguments as sbatch. The log goes to <logs>/<launcher stem>_%j.out (or
# _%A_%a.out for array jobs), where <logs> is the "logs" key of paths.json, and
# KLNN_REPO_ROOT is passed to the job so launchers find the repo from anywhere.
# Examples: ./submit.sh --array=0-9 data_generate/generate_simulator_v3.slurm
#           ./submit.sh --test-only arch/train_npe.slurm
set -euo pipefail

if (( $# < 1 )); then
    sed -n '2,10p' "$0" >&2
    exit 2
fi

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "${KLNN_REPO_ROOT}/klnn_paths.sh"
klnn_sbatch "$@"
