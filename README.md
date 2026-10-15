# PrOntoQA v0 on HiPerGator

本项目的运行目标是 **HiPerGator Linux CPU 作业**。复用已有环境，不创建 Conda 环境、不安装额外包，
不运行模型或训练。科学配置、基础世界、配对规则、base_id/example_id 和评估输入去重规则保持不变。

服务器路径统一配置在 [`configs/hipergator.env`](configs/hipergator.env)：

| 用途 | 默认路径 |
|---|---|
| 项目根目录 | `/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism` |
| 仓库 | `/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism/code` |
| 已有环境 | `/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism/envs/opd` |
| 数据运行目录的父目录 | `/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism/data/prontoqa_v0` |
| 日志父目录 | `/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism/logs` |

## 1. 首次克隆并核对固定子模块

以下命令由你在 HiPerGator 登录节点手动执行。登录节点只用于环境检查和提交作业。
若 `code` 已存在，使用下一节更新流程，不重复克隆。

```bash
PROJECT_ROOT=/blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism
mkdir -p "$PROJECT_ROOT"
git clone --recurse-submodules https://github.com/JingyangHe/Project1014.git "$PROJECT_ROOT/code"
cd "$PROJECT_ROOT/code"
git submodule sync --recursive
git submodule update --init --recursive --checkout
test "$(git -C third_party/prontoqa rev-parse HEAD)" = 066f73fc70bfaa92ed2476fd3c41b4951ad6e63a
git status --short
```

子模块的 Git gitlink 固定上述提交。不要运行 `git submodule update --remote`。
适配器以仓库路径定位上游，在导入期间临时切换到子模块目录读取 `bad_patterns.txt`，随后恢复原目录和 RNG 状态；
不修改子模块源文件。

## 2. 更新已有仓库

先处理本地改动，并确认没有正在使用此 checkout 的作业，再执行：

```bash
cd /blue/fsu-compsci-dept/jh24o.fsu/opd-mechanism/code
git status --short
git pull --ff-only origin main
git submodule sync --recursive
git submodule update --init --recursive --checkout
test "$(git -C third_party/prontoqa rev-parse HEAD)" = 066f73fc70bfaa92ed2476fd3c41b4951ad6e63a
```

## 3. 检查已有 Conda 环境

```bash
source configs/hipergator.env
module load "$PRONTOQA_CONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PRONTOQA_CONDA_ENV"
python scripts/check_environment.py --expected-prefix "$PRONTOQA_CONDA_ENV"
```

检查必须返回 `status: passed`：Python **3.12.15**、NumPy **2.2.6**、SciPy **1.15.3**，
且 Python prefix 必须是上述完整环境路径。最小 Python 数据依赖见
[`requirements-prontoqa-v0.txt`](requirements-prontoqa-v0.txt)；其余使用标准库。
还需要系统 Git、Bash 和 Slurm。检查脚本只读取版本，不修改环境。

## 4. 提交 CPU smoke 作业

```bash
submission=$(bash scripts/submit_prontoqa.sh smoke)
printf '%s\n' "$submission"
SMOKE_RUN=$(printf '%s\n' "$submission" | sed -n 's/^Run directory: //p')
PRONTOQA_SMOKE_JOB=$(cat "$SMOKE_RUN/slurm_job_id")
printf 'smoke run: %s\njob: %s\n' "$SMOKE_RUN" "$PRONTOQA_SMOKE_JOB"
```

提交入口为 [`scripts/submit_prontoqa.sh`](scripts/submit_prontoqa.sh)，CPU 作业为
[`scripts/prontoqa_cpu.sbatch`](scripts/prontoqa_cpu.sbatch)。请求 account/qos `fsu-compsci-dept`、
1 task、4 CPUs、16G、4 小时；不指定 partition，不申请 GPU，使用默认 CPU 分区。
脚本在 `sbatch` 前创建唯一日志目录，通过 `sbatch --output/--error` 传入动态路径。

每次生成在数据父目录下创建独立的 `smoke-UTC时间-随机后缀` 或 `full-UTC时间-随机后缀`：

```text
<run>/config.json                          提交时冻结的科学配置
<run>/slurm_job_id, job_info.txt            job ID、节点、代码版本与配置
<run>/environment.json, job_status.txt      环境检查与最终退出状态
<run>/selftest/                            逻辑与部署单元测试
<run>/data/                                本次主数据和 manifest
<run>/validation/                          独立验证报告
<run>/reproduction/                        同配置/种子的重复生成数据
<run>/repro-checks/                        逐字节复现比较报告
```

smoke/full 作业依次运行环境检查、selftest、生成、独立验证和重复生成检查。
任一步失败即非零退出。所有 BLAS/OpenMP/NumExpr 线程上限取实际 `SLURM_CPUS_PER_TASK`。

## 5. 查看队列、日志和报告

以下变量接续第 4 节；重新登录后可根据提交输出重新设置 `SMOKE_RUN`。

```bash
squeue -j "$PRONTOQA_SMOKE_JOB"
sacct -j "$PRONTOQA_SMOKE_JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 80 "$PRONTOQA_LOG_ROOT/$(basename "$SMOKE_RUN")/"*.out
tail -n 80 "$PRONTOQA_LOG_ROOT/$(basename "$SMOKE_RUN")/"*.err
```

等待作业结束后，确认 `COMPLETED`、`ExitCode=0:0`，再检查：

```bash
cat "$SMOKE_RUN/job_status.txt"
python - "$SMOKE_RUN" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
for relative in (
    'selftest/selftest_report.json',
    'data/validation_report.json',
    'validation/independent_validation_report.json',
    'repro-checks/reproducibility_report.json',
):
    report = json.loads((run / relative).read_text(encoding='utf-8'))
    assert report['status'] == 'passed', relative
    print(relative, report['status'])
manifest = json.loads((run / 'data/manifest.json').read_text(encoding='utf-8'))
assert manifest['counts'] == {'families': 100, 'core_records': 400, 'diagnostic_records': 0}
print('counts:', manifest['counts'])
print('project:', manifest['project'])
PY
```

## 6. smoke 通过后提交正常规模生成

确认上一节所有检查成功后执行：

```bash
bash scripts/submit_prontoqa.sh full
```

默认科学配置仍为 [`configs/prontoqa_v0.json`](configs/prontoqa_v0.json)：hops 1–5，
每种 hops 的 smoke/calibration/heldout 分别为 20/200/200 族。
正常规模共 2,100 族、8,400 条核心记录、2,000 条 calibration 诊断记录。
仅输出校准/留出评估数据及必要 smoke 检查，不导出训练集。
heldout 仅执行自动完整性检查，不用于模型或难度选择。

## 显式路径与仅验证

`--config` 支持仓库相对路径或绝对路径；提交时复制为 `<run>/config.json`，避免排队期间配置变化。
数据父目录和日志父目录可通过环境变量或以下参数覆盖；`--output` 可指定一个全新的数据目录。

```bash
bash scripts/submit_prontoqa.sh smoke --config configs/prontoqa_v0.json \
  --data-root "$PRONTOQA_DATA_ROOT" --log-root "$PRONTOQA_LOG_ROOT"

# 不生成数据；为既有数据创建一个新的验证运行目录和新的报告。
bash scripts/submit_prontoqa.sh validate \
  --config "$SMOKE_RUN/config.json" --output "$SMOKE_RUN/data"
```

在已经分配的 CPU 计算节点中，也可直接调用相同 Python 入口：

```bash
mkdir -p "$PRONTOQA_DATA_ROOT"
RUN=$(mktemp -d "$PRONTOQA_DATA_ROOT/manual-XXXXXX")
cp configs/prontoqa_v0.json "$RUN/config.json"
python -m prontoqa_v0 selftest --config "$RUN/config.json" --output "$RUN/selftest"
python -m prontoqa_v0 generate --stage smoke --config "$RUN/config.json" --output "$RUN/data"
python -m prontoqa_v0 validate --config "$RUN/config.json" --output "$RUN/data" --report-dir "$RUN/validation"
python -m prontoqa_v0 reproduce --config "$RUN/config.json" --output "$RUN/data" \
  --scratch "$RUN/reproduction" --report-dir "$RUN/repro-checks"
```

`generate/selftest` 拒绝已存在的输出目录，包括空目录；`reproduce` 也要求全新的 scratch。
`validate` 不修改数据，把报告写入新的 `--report-dir`；不指定时自动分配独立报告目录。
保留显式的 `export-eval` 旧数据元数据升级命令，它是唯一有意原位更新原记录导出的操作；详见数据说明。
不要设置 `PYTHONOPTIMIZE` 或使用 `python -O`。

## manifest、复现与检查边界

`data/manifest.json` 记录种子、配置及 SHA-256、上游提交和文件哈希、项目提交与 dirty 状态、
依赖版本、记录/族/唯一输入数量和数据文件哈希。JSONL 使用 UTF-8、LF、固定 JSON 序列化。
`execution.json`、`job_info.txt` 等单独记录时间、机器、Slurm allocation 和解释器路径，
不混入输入编号或数据哈希。重复生成检查对全部 JSONL 以及评估 manifest 做精确比较。

提交代码只包含代码、配置、测试、文档和小型固定测试样例。生成数据、日志、缓存与环境被 `.gitignore` 排除。
原历史 `data/smoke/2hop_random_seed0.json` 和新的 `tests/fixtures/v0-smoke-h5-0008.jsonl` 保留。
数据语义、比较关系和模型输入白名单见 [`README.prontoqa_v0.md`](README.prontoqa_v0.md)。

本地检查状态见 [`docs/local_checks.md`](docs/local_checks.md)。本地 Python 测试、生成和 Bash 语法检查通过
不等于 HiPerGator 作业已运行；Conda module、account/qos 权限、默认分区和 Slurm 调度需由服务器 smoke 实测。
没有在本轮自动登录服务器或提交服务器作业。

官方操作依据：[Conda 脚本环境](https://docs.rc.ufl.edu/software/conda_using_environments/)、
[按路径激活环境](https://docs.rc.ufl.edu/software/conda_installing_packages/)、
[Slurm 调度](https://docs.rc.ufl.edu/quickstart/scheduling/)。
