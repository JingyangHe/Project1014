#!/usr/bin/env bash
# Submit from any directory. Server paths live in configs/hipergator.env.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "${repo_root}/configs/hipergator.env"

usage() {
    cat <<'EOF'
Usage: bash scripts/submit_prontoqa.sh {smoke|full|validate} [options]
  --config FILE      Scientific JSON config (relative to repository or absolute)
  --output DIR       New dataset directory; required existing directory for validate
  --data-root DIR    Parent for unique run directories (default: configured /blue path)
  --log-root DIR     Log parent (default: configured /blue path)
Use PRONTOQA_CONDA_ENV to override the configured existing Conda environment.
EOF
}

if [[ $# -eq 0 || ${1:-} == --help ]]; then usage; exit 0; fi
mode=$1
shift
case "$mode" in smoke|full|validate) ;; *) usage >&2; exit 2 ;; esac
config="${repo_root}/configs/prontoqa_v0.json"
output=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config=${2:?--config requires a file}; shift 2 ;;
        --output) output=${2:?--output requires a directory}; shift 2 ;;
        --data-root) PRONTOQA_DATA_ROOT=${2:?--data-root requires a directory}; shift 2 ;;
        --log-root) PRONTOQA_LOG_ROOT=${2:?--log-root requires a directory}; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $config == /* ]] || config="${repo_root}/${config}"
config=$(realpath -- "$config")
[[ -f $config ]] || { printf 'Config not found: %s\n' "$config" >&2; exit 2; }
if [[ -n $output ]]; then output=$(realpath -m -- "$output"); fi
if [[ $mode == validate ]]; then
    [[ -n $output && -f $output/validation_report.json ]] || {
        printf 'validate requires --output pointing to an existing generated dataset\n' >&2; exit 2;
    }
elif [[ -n $output && -e $output ]]; then
    printf 'Refusing existing output directory: %s\n' "$output" >&2
    exit 2
fi
command -v sbatch >/dev/null || { printf 'sbatch is unavailable; run this on HiPerGator\n' >&2; exit 2; }

# Slurm opens stdout/stderr before the worker starts: create log directories now.
mkdir -p -- "$PRONTOQA_DATA_ROOT" "$PRONTOQA_LOG_ROOT"
run_dir=$(mktemp -d "${PRONTOQA_DATA_ROOT}/${mode}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
log_dir="${PRONTOQA_LOG_ROOT}/$(basename -- "$run_dir")"
mkdir -p -- "$log_dir"
[[ -n $output ]] || output="${run_dir}/data"
cp -- "$config" "${run_dir}/config.json"
job_id=$(sbatch --parsable --chdir="$repo_root" --export=ALL \
    --job-name="prontoqa-${mode}" --output="${log_dir}/%x-%j.out" --error="${log_dir}/%x-%j.err" \
    "${repo_root}/scripts/prontoqa_cpu.sbatch" "$repo_root" "$mode" "$run_dir" \
    "${run_dir}/config.json" "$output")
printf '%s\n' "$job_id" > "${run_dir}/slurm_job_id"
printf 'Job ID: %s\nRun directory: %s\nDataset: %s\nLogs: %s\n' "$job_id" "$run_dir" "$output" "$log_dir"
