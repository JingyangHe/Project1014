# PrOntoQA 衍生校准数据集 v0

本实现复用 `third_party/prontoqa` 的生成器、逻辑公式和证明，固定提交
`066f73fc70bfaa92ed2476fd3c41b4951ad6e63a`。不修改子模块；只运行 CPU 数据构造和自动检查。
使用现有环境：Python **3.12.15**、NumPy **2.2.6**、SciPy **1.15.3**。

## 从项目根目录运行

HiPerGator 的克隆、环境检查和 smoke/full/validate 提交流程见 [README.md](README.md)。
Linux CLI 要求显式提供输出目录，避免覆盖已有结果。在已分配的 CPU 计算节点中：

```bash
source configs/hipergator.env
mkdir -p "$PRONTOQA_DATA_ROOT"
RUN=$(mktemp -d "$PRONTOQA_DATA_ROOT/manual-XXXXXX")
cp configs/prontoqa_v0.json "$RUN/config.json"
python -m prontoqa_v0 selftest --config "$RUN/config.json" --output "$RUN/selftest"
python -m prontoqa_v0 generate --stage smoke --config "$RUN/config.json" --output "$RUN/data"
python -m prontoqa_v0 validate --config "$RUN/config.json" --output "$RUN/data" --report-dir "$RUN/validation"
python -m prontoqa_v0 reproduce --config "$RUN/config.json" --output "$RUN/data" --scratch "$RUN/reproduction" --report-dir "$RUN/repro-checks"
```

对已经生成的 v0 数据，只添加编号、比较关系和评估导出，不重新抽取基础世界：

```bash
python -m prontoqa_v0 export-eval --config "$RUN/config.json" --output "$RUN/data" --report-dir "$RUN/metadata-upgrade"
```

`export-eval` 保留所有原记录 ID、版本、题干、查询、标签、证明、逻辑和种子；
报告中保存升级前后原始字段的 SHA-256，验证只有新编号和诊断比较字段被添加。

所有路径相对项目根目录解析；上游导入时由适配器临时切换工作目录，解决
`bad_patterns.txt` 的相对路径读取问题，并在结束或异常时恢复目录和 Python/NumPy 随机状态。
Git 所有权检查使用单次命令的 `safe.directory` 参数，不改全局 Git 配置。
不要使用 Python `-O`，验证命令会拒绝禁用断言的运行方式。

科学配置位于 `configs/prontoqa_v0.json`；服务器路径位于 `configs/hipergator.env`。
每次作业在服务器数据父目录下创建独立的运行目录，具体数据位于 `<run>/data/`。
`generate --stage full` 会先重新生成并验证 smoke，再生成其余数据。
每个 pool/hops 最多尝试 30,000 次；达不到数量时非零退出并写入
`generation_failure.json`，不放宽限制。成功导出前完成所请求数据的全部检查。

## 数量与文件

| pool | 每种 hops 的题目族 | 每种 hops 的核心记录 | 合计题目族 | 合计核心记录 |
|---|---:|---:|---:|---:|
| smoke | 20 | 80 | 100 | 400 |
| calibration | 200 | 800 | 1,000 | 4,000 |
| heldout | 200 | 800 | 1,000 | 4,000 |

支持 hops 1–5。每族四个核心版本：`A_true`、`B_true`、`A_false`、`B_false`。
每种 hops 从 calibration 固定抽取 50 族，对每个核心版分别生成
`__key_flip` 和 `__irrelevant_flip`：共 250 族、2,000 条诊断记录。
总计 **2,100 个题目族、8,400 条核心记录、2,000 条诊断记录**。

- `smoke.jsonl`、`calibration.jsonl`、`heldout.jsonl`：分池核心数据。
- `diagnostics.jsonl`：诊断数据，`pool` 仍为 `calibration`，保留族归属和父记录。
- `provenance.jsonl`：每族原始上游文本、TPTP 公式、证明 AST 引用、原始 IR、种子及适配映射。
- `rejections.jsonl`：每次拒绝的候选种子、类型和原因。
- `manifest.json`、`validation_report.json`：数量、分布、变换检查、配置、依赖、项目提交与 dirty 状态、上游及数据文件哈希。
- `<run>/validation/independent_validation_report.json`：重新读盘、回解析、来源证明及变换重放的结果。
- `smoke_gate_report.json`、`<run>/selftest/selftest_report.json`、`<run>/repro-checks/reproducibility_report.json`：门禁、测试和逐字节复现结果。
- 各命令的 `execution.json`：机器、时间、Slurm allocation 等运行信息，与数据 manifest 分离。
- `config.snapshot.json`：实际配置；`examples.md`：仅从 smoke 选取的五个完整配对题目族。

JSONL 使用 UTF-8、LF 换行，每行一个对象。记录含 `id/base_id/pool/hops/kind/variant`、
名字与组、`question/query/answer`、`logic`、`proof`、`render_plan`、精确变换记录及来源引用。
事实与每条证明结论都显式包含实体名字。规则 ID 及事实 ID 在族内稳定，证明步骤引用这些 ID。

`base_id` 标识题目族；原 `id` 仍唯一标识一条带来源的记录。
新增 `example_id` 标识精确输入：对仅包含 `question/query` 的 JSON 对象使用
`ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False` 序列化，
再对其 UTF-8 字节计算 SHA-256。没有名字、空白或 Unicode 归一化；名字、规则文字、
句序或查询改变时编号相应改变。答案、证明、版本及来源不参与编号；同编号标签冲突必须失败。

## 唯一评估输入与比较关系

- `evaluation_inputs.jsonl`：唯一 `{example_id, prompt}` 清单。
- `evaluation_record_map.jsonl`：每条原记录到 example_id 的映射，以及它在比较中的 source/result 角色。
- `evaluation_comparisons.jsonl`：名字比较、单关键规则翻转、单无关规则翻转，以及单独标记的核心双规则交换。
- `evaluation_manifest.json`：输入定义、指令哈希、实际记录/唯一输入/题目族/比较数量、验证状态及导出文件哈希。

每条诊断记录新增 `comparison`：含来源核心记录、来源和结果 example_id、结果记录、修改类型、
实际规则的修改前后表示，以及 `expected_answer_relation`（`flip` 或 `same`）。
名字比较为 A_true→B_true、A_false→B_false，预期标签保持；核心 True→False 比较明确记录两条修改。
所有原记录都保留。唯一输入回答可通过映射关联回多条诊断比较；去重不用于训练采样，也不改变其频率或权重。

`inspection_records.jsonl` 单独保存 smoke 案例 `v0-smoke-h5-0008` 的 4 个核心与 8 个诊断版本，
配套 `inspection_inputs.jsonl`、`inspection_record_map.jsonl`、`inspection_comparisons.jsonl`。
该案例应为 12 条记录、8 个精确输入；它不增加主数据池计数。其他族按实际内容统计。

发送给模型时只取 `evaluation_inputs.jsonl` 中的 **prompt 字符串**。
不同模型、不同训练检查点必须分别运行评估；回答缓存至少按
`(model_id, checkpoint_id, instruction_sha256, example_id)` 区分，禁止跨模型或检查点复用。
本轮仅导出输入及映射，没有调用模型。

## 逻辑和配对约束

单实体、一元类别包含、类别推出肯定或否定属性；所有前件均为肯定类别。
查询始终为肯定属性，例如 `True or false: Nora is green.`。
求解器计算全部正向闭包：事实代价为 0，每次规则应用代价为 1，反复松弛得到最短距离。
True 必须只可推出查询；False 必须只可推出其否定。未知、查询矛盾或其他可达属性矛盾均拒绝。
不使用逆否、闭世界假设或矛盾爆炸。上游的步数包含事实步骤，调用时使用 `hops + 1`，然后独立重算。

先验证上游实际证明 AST，再将公式转换为类型明确的 IR。
不支持的合取、析取、存在式、否定类别、多实体、非一元谓词等显式拒绝。
类别/属性的随机双射、名字替换、规则打乱和极性操作均作用于 IR，不依靠文本替换。
每族只抽取一个 A 名字及一个 B 名字，True/False 复用该名字对、句序和每条规则的句式。
每个版本均重新求解、生成最短证明、渲染和回解析。

目标必须恰有一个可达的决定规则和一个前件不可达、结论相反的无关规则。
删除关键规则必须使目标未知。核心 True→False 交换两条目标规则的极性，
精确记录 **两条** 修改；因此四个核心版规则数和 `not` 单词数相同。
诊断版相对其 `parent_id` 仅修改一条规则，分别验证答案改变/保持，并记录修改前后 IR。
所有版本均要求同一最短 hops、无冲突；诊断版不要求否定词数平衡。

去重签名移除实体、规则/事实 ID、句序及查询，对事实和规则排序去重；保留谓词拼写。
所有核心和可能的单规则干预世界在接纳题目族时一起预留签名，防止后续诊断引入泄漏。
相同世界只允许在同一个题目族内因名字版本或等价干预重复；跨族、跨池重复均拒绝。
heldout 仅作自动构造和完整性检查，未用于选择模型、超参数或经验难度。

## 模型输入与后续采样

```python
from prontoqa_v0.render import build_prompt
prompt = build_prompt(record)
```

该函数只通过白名单读取 `question/query`，组合已有固定指令，保留题目中的实体名字。
它不序列化整条记录；base_id、example_id、记录 ID、variant、名字组、hops、答案、证明、
变换、规则编号和 pool 等后台字段均不进入提示。原 `build_model_input(question, query)`
作为兼容入口调用同一实现。该函数只构造原始题目提示，不添加 OPD 学生生成前缀。

```python
from prontoqa_v0.sampling import sample_bias_pair
train_50, train_90 = sample_bias_pair(
    independent_training_core_records,
    seed=42,
    correlations=(0.5, 0.9),
    true_name_group='A',  # 也支持反向：'B'
)
```

此函数只接受 **独立 training 池** 的完整核心题目族；拒绝 smoke、calibration、heldout 和诊断记录。
每族取一个版本，两份输出共享族和 True/False 选择，只改变名字组分配。
每个 hops 内标签均衡，每个 hops/标签内严格控制相关比例，保留反向名字—标签组合。
比例不能精确实现时直接报错，不四舍五入。50%/90% 配对要求每个 hops 至少 20 族，
且族数为 20 的倍数。方向由 `true_name_group` 配置，False 使用另一组。
配置中的 `sampling` 给出默认参数，调用方可按此传入。
本轮只测试该函数，**没有导出任何训练集**。

## 模块与已知限制

`upstream` 负责适配；`logic` 求解；`transform` 变换；`render` 渲染/解析/模型输入；
`dataset` 导出；`validation` 与 `audit` 检查当前数据和上游来源；`sampling` 实现配对采样。
测试涵盖已知真/假/未知/矛盾、零步/短路/循环、篡改证明或文本、泄漏、复现和最大尝试限制。

这是受限的合成推理基准：上游虚构类别有树/链结构偏好；渲染只用三种受控英语句式。
hops 是逻辑最短规则步数，不代表模型实测难度。本轮不调用模型、不使用 GPU。
名字组是显式配置的代理分组，不能据此作人口属性或公平性推断。
去重保留谓词拼写，因此不同词汇重命名但结构同构的世界仍可能存在。
不支持多实体、关系谓词、合取/析取、存在推理、逆否或任意自然语言解析。

上游引用：Saparov & He (ICLR 2023), *Language Models Are Greedy Reasoners:
A Systematic Formal Analysis of Chain-of-Thought*；原许可证保留于 `third_party/prontoqa/LICENSE`。
