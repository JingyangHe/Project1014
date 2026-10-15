#!/usr/bin/env bash
# Offline submission contract test: no Slurm, Conda or server connection.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mkdir -p -- "${repo_root}/work"
sandbox=$(mktemp -d "${repo_root}/work/submit-test-XXXXXX")
mkdir -p -- "${sandbox}/bin"
cat > "${sandbox}/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    case "$arg" in
        --output=*|--error=*) [[ -d $(dirname -- "${arg#*=}") ]] || exit 61 ;;
    esac
done
printf '%s\n' "$@" > "$SBATCH_CAPTURE"
printf '12345\n'
STUB
chmod +x "${sandbox}/bin/sbatch"
export PATH="${sandbox}/bin:${PATH}"
export PRONTOQA_DATA_ROOT="${sandbox}/runs" PRONTOQA_LOG_ROOT="${sandbox}/logs"
export SBATCH_CAPTURE="${sandbox}/arguments.txt"

for mode in smoke full; do
    result=$(bash "${repo_root}/scripts/submit_prontoqa.sh" "$mode")
    [[ $result == *'Job ID: 12345'* ]]
    mapfile -t args < "$SBATCH_CAPTURE"
    count=${#args[@]}
    [[ ${args[count-4]} == "$mode" ]]
    run_dir=${args[count-3]}
    [[ -f $run_dir/config.json && -f $run_dir/slurm_job_id ]]
    [[ ${args[count-1]} == "$run_dir/data" ]]
    cmp -- "${repo_root}/configs/prontoqa_v0.json" "$run_dir/config.json"
done
existing="${sandbox}/existing dataset"
mkdir -p -- "$existing"
printf '{}\n' > "$existing/validation_report.json"
bash "${repo_root}/scripts/submit_prontoqa.sh" validate --output "$existing" > "${sandbox}/validate.txt"
mapfile -t args < "$SBATCH_CAPTURE"
count=${#args[@]}
[[ ${args[count-4]} == validate && ${args[count-1]} == "$existing" ]]
cp -- "$SBATCH_CAPTURE" "${sandbox}/before.txt"
if bash "${repo_root}/scripts/submit_prontoqa.sh" full --output "$existing" > "${sandbox}/refusal.txt" 2>&1; then
    printf 'Existing output was unexpectedly accepted\n' >&2
    exit 1
fi
cmp -- "$SBATCH_CAPTURE" "${sandbox}/before.txt"
printf 'PASS: smoke/full/validate submission, log creation, quoting and overwrite guard; sandbox=%s\n' "$sandbox"
