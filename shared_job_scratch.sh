#!/bin/bash
# Shared scratch setup for KL-NN batch jobs.
#
# Source this file and call setup_shared_job_scratch STAGE once. The resulting
# per-task directory is exported through TMPDIR, TMP, and TEMP and is removed
# on normal exit, failure, or a catchable termination signal.

readonly KLNN_SHARED_TMP_ROOT="/ocean/projects/phy250048p/shared/tmp"

_cleanup_shared_job_scratch() {
    local status=$?
    trap - EXIT
    local scratch="${SHARED_JOB_TMPDIR:-}"
    if [[ -z "${scratch}" || "${scratch}" == "${KLNN_SHARED_TMP_ROOT}" || "${scratch}" != "${KLNN_SHARED_TMP_ROOT}"/klnn-* ]]; then
        echo "Refusing unsafe shared-scratch cleanup target: ${scratch:-<unset>}" >&2
        exit 97
    fi
    if [[ -e "${scratch}" ]] && ! rm -rf -- "${scratch}"; then
        echo "Failed to remove shared scratch directory: ${scratch}" >&2
        if (( status == 0 )); then
            status=98
        fi
    fi
    exit "${status}"
}

setup_shared_job_scratch() {
    if (( $# != 1 )); then
        echo "setup_shared_job_scratch requires exactly one stage label" >&2
        return 2
    fi
    local stage="$1"
    if [[ ! "${stage}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "Invalid shared-scratch stage label: ${stage}" >&2
        return 2
    fi
    local job_token="${SLURM_JOB_ID:-manual-$$}"
    local array_token="${SLURM_ARRAY_TASK_ID:-none}"

    mkdir -p -- "${KLNN_SHARED_TMP_ROOT}"
    SHARED_JOB_TMPDIR="${KLNN_SHARED_TMP_ROOT}/klnn-${stage}-${job_token}-${array_token}"
    if [[ -e "${SHARED_JOB_TMPDIR}" ]]; then
        echo "Refusing to reuse shared scratch directory: ${SHARED_JOB_TMPDIR}" >&2
        return 2
    fi
    mkdir -m 700 -- "${SHARED_JOB_TMPDIR}"
    export SHARED_JOB_TMPDIR
    export TMPDIR="${SHARED_JOB_TMPDIR}"
    export TMP="${SHARED_JOB_TMPDIR}"
    export TEMP="${SHARED_JOB_TMPDIR}"

    trap _cleanup_shared_job_scratch EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo "shared scratch=${SHARED_JOB_TMPDIR}"
}
