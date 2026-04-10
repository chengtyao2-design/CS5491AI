# TSPLib 标准测试实例 & 实验配置参考

本目录包含 TSP 标准测试文件，用于 FunSearch 求解器 benchmark。支持 EUC_2D、ATT、GEO、EXPLICIT 等格式（通过 tsplib95）。

---

## 目录结构

- **根目录**：`eil51.tsp`、`berlin52.tsp`，含手动添加的 OPTIMUM 行，可直接计算 gap
- **tsp/**：完整 TSPLib Symmetric 实例集（来自 [Rice TSPLib](https://softlib.rice.edu/pub/tsplib/tsp/)），含 `.tsp` 与 `.opt.tour` 标准解

## 推荐实例

| 文件 | 城市数 | 距离类型 | 最优解来源 |
|------|--------|----------|------------|
| eil51.tsp | 51 | EUC_2D | 根目录含 OPTIMUM；tsp/ 含 eil51.opt.tour |
| berlin52.tsp | 52 | EUC_2D | 根目录含 OPTIMUM；tsp/ 含 berlin52.opt.tour |
| att48.tsp | 48 | ATT | tsp/ 含 att48.opt.tour |
| gr17.tsp | 17 | EXPLICIT | tsp/ |
| ulysses22.tsp | 22 | GEO | tsp/ 含 ulysses22.opt.tour |

## Gap 计算说明

- 含 **OPTIMUM**、**BEST_KNOWN** 或 **OPTIMAL_VALUE** 的 `.tsp` 可直接计算 gap
- 若 `.tsp` 无上述字段，会自动查找同目录下的 `.opt.tour`（如 `eil51.tsp` → `eil51.opt.tour`），解析 tour 并用距离矩阵计算 optimal
- 启动时日志会提示每个实例的 optimal 来源（`.tsp` / `.opt.tour`），若无 optimal 会给出 WARNING，gap 相关图表将不生成
- 使用 TSPLib 且有 optimal 时，结果目录会额外生成 `optimal_comparison.png`（各实例 gap 柱状图）和 `gap_progression.png`（gap 随迭代变化曲线）

## 使用方式

```bash
# 使用 TSPLib 实例（指定 --tsplib 时仅用 tsplib，不叠加随机）
python run.py --problem tsp --tsplib data/tsplib/eil51.tsp data/tsplib/berlin52.tsp
python run.py --problem tsp --tsplib data/tsplib/tsp/eil51.tsp data/tsplib/tsp/att48.tsp data/tsplib/tsp/berlin52.tsp

# 使用随机实例（--random 城市数，--seed 可选，数量须与 --random 一致）
python run.py --problem tsp --random 15 20 25
python run.py --problem tsp --random 15 20 25 --seed 42 123 456

# 不指定 --tsplib 和 --random 时，使用预设：3 个随机实例 (15,20,25) + seed (42,123,456)
python run.py --problem tsp
```

## 结果输出

结果保存在 `result/tsp_{unix_timestamp}/` 下：

| 文件 | 内容 |
|------|------|
| `score_progression.png` | 全局最佳分数随迭代变化曲线 |
| `final_results.json` | 迭代数、token、最佳分数、效率指标等统计 |
| `best_program.py` | 当前最优程序代码 |
| `experiment_summary.md` | 实验摘要（含 Sample Efficiency 章节） |
| `optimal_comparison.png` | 各实例 gap 柱状图（有 optimal 时生成） |
| `gap_progression.png` | gap 随迭代变化曲线（有 optimal 时生成） |

---

## CLI 参数完整列表

所有参数均通过 `python run.py` 传入。

### 问题与实例

| 参数 | 说明 | 默认 |
|------|------|------|
| `--problem` | 问题类型：`tsp` 或 `admissible` | `admissible` |
| `--tsplib PATH [PATH ...]` | TSPLib `.tsp` 文件路径，可多个；指定后忽略 `--random` | 无（使用随机实例） |
| `--random N [N ...]` | 随机实例城市数，可多个（如 `15 20 25`） | 无 |
| `--seed S [S ...]` | 随机实例 seed，数量须与 `--random` 一致；未指定时默认 seed=0 | 无 |

### 样本效率开关

| 参数 | 说明 | 默认行为 |
|------|------|----------|
| `--no-functional-dedup` | 强制禁用功能级去重（覆盖 config） | 依 config（当前默认 False） |
| `--progressive-eval` / `--no-progressive-eval` | 渐进式评估开关 | 依 config（当前默认 False） |
| `--adaptive-sampling` / `--no-adaptive-sampling` | 自适应采样开关 | 依 config（当前默认 False） |
| `--weighted-island` / `--no-weighted-island` | 加权岛屿选择开关 | 依 config（当前默认 False） |
| `--feedback-in-prompt` / `--no-feedback-in-prompt` | 失败反馈注入 prompt 开关 | 依 config（当前默认 False） |

> **注**：`--no-functional-dedup` 是单向标志（只能关闭），其他效率项均支持 `--feature` / `--no-feature` 互斥组，未传时以 `config.py` 中的默认值为准。

---

## 核心配置项（implementation/config.py）

通过修改 `implementation/config.py` 中的 `Config` 和 `ProgramsDatabaseConfig` 调整。

### Config（顶层）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `iterations` | 最大迭代轮数 | `50` |
| `num_samplers` | Sampler 数量（串行时仅第一个工作） | `1` |
| `num_evaluators` | Evaluator 数量 | `140` |
| `samples_per_prompt` | 每轮每个 prompt 的基础采样数 | `4` |
| `early_stop_patience` | 无改进轮数阈值触发早停；`-1` 禁用 | `-1` |
| `result_dir` | 结果输出根目录 | `"result"` |
| `problem` | 问题名称，用于结果子目录命名 | `"admissible"` |
| `goal_description` | 写入 LLM prompt 的问题目标描述 | （各问题自动设置） |

### ProgramsDatabaseConfig（programs_database 子配置）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `functions_per_prompt` | 每个 prompt 中包含的历史程序数量 | `2` |
| `num_islands` | 岛屿数量（多样性机制） | `10` |
| `reset_period` | 弱岛屿重置周期（秒） | `600` |
| `cluster_sampling_temperature_init` | Cluster 采样 softmax 初始温度 | `0.1` |
| `cluster_sampling_temperature_period` | 温度周期衰减的程序注册数周期 | `50` |

---

## 样本效率配置

以下选项全部**默认关闭**，可通过 CLI 参数或直接修改 `config.py` 开启。

### 功能级去重（Functional Deduplication）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `functional_dedup` | 启用功能级去重（Tier1 语法 hash + Tier2 执行输出比对） | `False` |
| `dedup_tier1_only` | 仅做 Tier1 语法 hash 去重，跳过 Tier2 执行比对 | `False` |

- **Tier1**：对程序代码做 hash，语法完全相同则跳过，速度极快
- **Tier2**（仅 TSP）：在相同实例上运行程序，比对 tour 签名；功能等价但语法不同的程序也会被跳过

### 渐进式评估（Progressive Evaluation）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `progressive_eval` | 启用两阶段评估：先用小实例快速筛选，通过后再完整评估 | `False` |
| `stage1_timeout` | Stage1 单次评估超时秒数 | `5` |
| `stage1_score_threshold_pct` | Stage1 通过所需最低分数阈值（相对当前最优的百分比）；`0.0` 表示仅需执行不崩溃 | `0.0` |

### 自适应采样（Adaptive Sampling）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `adaptive_sampling` | 启用自适应采样：连续无改进时减少每轮采样数以节省 token | `False` |
| `min_samples_per_prompt` | 自适应模式下每轮最少采样数 | `2` |
| `max_samples_per_prompt` | 自适应模式下每轮最多采样数（上限） | `4` |
| `reduce_after_no_improve` | 连续无改进多少轮后触发采样数减少 | `3` |

### 加权岛屿选择（Weighted Island Sampling）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `weighted_island_sampling` | 启用按岛屿历史最优分数加权选择岛屿（否则均匀随机） | `False` |
| `island_sampling_temperature` | 加权选岛 softmax 温度；越大越均匀，越小越偏向高分岛屿 | `1.0` |

### 失败反馈注入（Feedback in Prompt）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `feedback_in_prompt` | 将近期被拒绝/失败的程序摘要注入 LLM prompt，引导模型避免重复错误 | `False` |

---

## 动态温度配置

FunSearch 内部有三层采样，各自的温度机制如下：

```
get_prompt()
  ├── 1. 选岛：均匀随机 或 softmax(best_score_per_island, island_sampling_temperature)
  ├── 2. 选 Cluster：softmax(cluster_scores, T_dynamic)  ← 周期衰减 + 可自适应升温
  └── 3. 选具体程序：softmax(-normalized_length, T=4.0 固定)
```

### Cluster 级温度（周期衰减）

温度从 `cluster_sampling_temperature_init` 开始，每注册一个程序线性衰减，每隔 `cluster_sampling_temperature_period` 个注册程序重置一次：

```
temperature = init × (1 - (num_programs % period) / period)
```

| 配置项 | 位置 | 说明 | 默认值 |
|--------|------|------|--------|
| `programs_database.cluster_sampling_temperature_init` | `ProgramsDatabaseConfig` | Cluster 采样初始温度（越高探索越随机） | `0.1` |
| `programs_database.cluster_sampling_temperature_period` | `ProgramsDatabaseConfig` | 温度重置周期（程序注册数为单位）；应与实验规模匹配 | `50` |

> **注**：原始 FunSearch 默认值为 `30000`，远超小规模实验（通常注册 40~200 个程序），导致温度永不重置，退化为单调退火。当前已修正为 `50`。

### Island 级温度（加权选岛）

| 配置项 | 位置 | 说明 | 默认值 |
|--------|------|------|--------|
| `weighted_island_sampling` | `Config` | 是否按岛屿分数加权选择（否则均匀随机） | `False` |
| `island_sampling_temperature` | `Config` | 加权选岛 softmax 温度；`1.0` 为标准 softmax，增大趋于均匀，减小趋于贪心 | `1.0` |

### 自适应升温（Adaptive Reheat）

当实验连续多轮无全局改进时，自动提高 Cluster 采样温度以跳出局部最优：

| 配置项 | 位置 | 说明 | 默认值 |
|--------|------|------|--------|
| `adaptive_temperature` | `Config` | 启用自适应升温机制 | `False` |
| `reheat_after_no_improve` | `Config` | 连续无改进多少轮后触发升温 | `5` |
| `reheat_temperature_multiplier` | `Config` | 升温倍数（实际温度 = init × multiplier × 衰减因子） | `3.0` |

---

## 快速参考：全部开启效率优化的运行命令

```bash
# TSPLib 实例 + 全部效率优化
python run.py --problem tsp --tsplib data/tsplib/eil51.tsp \
  --progressive-eval --adaptive-sampling --weighted-island --feedback-in-prompt

# 随机实例 + 仅功能级去重（在 config.py 中设置 functional_dedup=True）
python run.py --problem tsp --random 15 20 25 --seed 42 123 456
```

> `functional_dedup` 和动态温度相关选项（`adaptive_temperature`、`island_sampling_temperature` 等）目前只能在 `implementation/config.py` 中修改，暂无对应 CLI 参数。
