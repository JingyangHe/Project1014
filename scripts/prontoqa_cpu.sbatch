#!/bin/bash -l
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# No partition/GPU request: use the site's default CPU partition.
# Submit through submit_prontoqa.sh so log paths already exist.
set -euo pipefail
repo_root=${1:?repository directory required}
mode=${2:?smoke/full/validate required}
run_dir=${3:?reserved run directory required}
config=${4:?scientific config required}
output=${5:?dataset directory required}
cd -- "$repo_root"
source "${repo_root}/configs/hipergator.env"

finish() {
    code=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" > "${run_dir}/job_status.txt"
    exit "$code"
}
trap finish EXIT
threads=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set by Slurm}
[[ $threads =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid CPU allocation\n' >&2; exit 2; }
export OMP_NUM_THREADS=$threads OMP_THREAD_LIMIT=$threads OPENBLAS_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads NUMEXPR_NUM_THREADS=$threads NUMEXPR_MAX_THREADS=$threads
export BLIS_NUM_THREADS=$threads VECLIB_MAXIMUM_THREADS=$threads
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
[[ -z ${PYTHONOPTIMIZE:-} ]] || { printf 'PYTHONOPTIMIZE must be unset\n' >&2; exit 2; }

module load "$PRONTOQA_CONDA_MODULE"
# Conda activation scripts may reference unset optional variables.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PRONTOQA_CONDA_ENV"
set -u
python scripts/check_environment.py --expected-prefix "$PRONTOQA_CONDA_ENV" --report "${run_dir}/environment.json"
{
    printf 'started_utc=%s\nnode=%s\njob_id=%s\ncpus=%s\n' \
        "$(date -u +%FT%TZ)" "$(hostname)" "${SLURM_JOB_ID:?}" "$threads"
    printf 'commit=%s\nconfig=%s\ndataset=%s\n' "$(git rev-parse HEAD)" "$config" "$output"
    git status --short
    cat -- "$config"
} > "${run_dir}/job_info.txt"
cat -- "${run_dir}/job_info.txt"

case "$mode" in
    smoke|full)
        python -m prontoqa_v0 selftest --config "$config" --output "${run_dir}/selftest"
        python -m prontoqa_v0 generate --config "$config" --stage "$mode" --output "$output"
        python -m prontoqa_v0 validate --config "$config" --output "$output" --report-dir "${run_dir}/validation"
        python -m prontoqa_v0 reproduce --config "$config" --output "$output" \
            --scratch "${run_dir}/reproduction" --report-dir "${run_dir}/repro-checks"
        ;;
    validate)
        python -m prontoqa_v0 validate --config "$config" --output "$output" --report-dir "${run_dir}/validation"
        ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
