# FunSearch TSP Solver

基于 [FunSearch](https://github.com/google-deepmind/funsearch) 框架的旅行商问题（TSP）求解器。通过大语言模型（LLM）自动进化贪心构造启发式，在贪心插入框架下优化 TSP 路径长度。

## 功能特性

- **LLM 驱动的程序进化**：使用 LLM 生成并迭代改进 `priority` 启发式函数
- **多岛屿进化**：岛屿机制保持种群多样性，定期重置弱岛屿
- **早停机制**：当全局最佳分数连续多轮无改进时自动退出
- **结果可视化**：自动生成分数变化曲线图并保存到 `result/` 目录
- **支持 TSPLib**：可加载标准 TSPLib 格式实例
- **随机实例**：支持在单位正方形内生成随机欧氏 TSP 实例

## 项目结构

```
CS5491AI/
├── run.py                    # 主入口
├── .env.example              # 环境变量示例
├── .env                      # 实际配置（需自行创建，不提交）
├── implementation/
│   ├── funsearch.py          # FunSearch 主流程
│   ├── config.py             # 配置定义
│   ├── sampler.py            # LLM 采样与迭代控制
│   ├── evaluator.py          # 程序评估（沙箱执行）
│   ├── programs_database.py  # 程序库与岛屿管理
│   ├── code_manipulation.py  # 代码解析与操作
│   ├── tsp_utils.py          # TSP 工具（TSPLib、随机实例）
│   ├── specification_tsp.txt # TSP 问题规格
│   └── specification_nonsymmetric_admissible_set.txt  # Cap Set 规格
└── result/                   # 实验结果（按时间戳分目录）
    └── YYYYMMDD_HHMMSS/
        ├── score_progression.png   # 分数随迭代变化图
        ├── final_results.json      # 统计数据
        ├── best_program.py         # 最佳程序代码
        └── experiment_summary.md   # 实验摘要
```

## 环境要求

- Python 3.10+
- 依赖：`numpy`, `scipy`, `absl-py`, `openai`, `python-dotenv`
- 可选：`matplotlib`（用于生成分数曲线图）

## 安装

```bash
# 克隆项目
cd CS5491AI

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY 和 LLM_MODEL
```

## 配置

### 环境变量（.env）

| 变量 | 说明 | 示例 |
|------|------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | 从 [OpenRouter](https://openrouter.ai/) 获取 |
| `LLM_MODEL` | 使用的模型 | `arcee-ai/trinity-large-preview:free` |

若出现 404 "guardrail restrictions and data policy"，请前往 [OpenRouter 隐私设置](https://openrouter.ai/settings/privacy) 放宽数据保留策略。

### 运行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--problem` | 问题类型：`tsp` 或 `admissible` | `admissible` |
| `--tsplib` | TSPLib .tsp 文件路径（可多个） | 无，使用随机实例 |
| `--no-functional-dedup` | 禁用功能级重复检测 | 默认启用 |
| `--progressive-eval` | 启用渐进式评估（先小实例筛选） | 默认禁用 |
| `--adaptive-sampling` | 启用自适应采样（无改进时减少样本） | 默认禁用 |
| `--weighted-island` | 启用加权岛屿选择 | 默认禁用 |
| `--feedback-in-prompt` | 在 prompt 中注入失败样本反馈 | 默认禁用 |

### 配置项（implementation/config.py）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `iterations` | 最大迭代轮数 | 10 |
| `early_stop_patience` | 无改进轮数阈值，超过则早停；-1 禁用 | 5 |
| `num_samplers` | Sampler 数量（串行时仅第一个工作） | 1 |
| `num_evaluators` | Evaluator 数量 | 140 |
| `samples_per_prompt` | 每轮每个 prompt 的采样数 | 4 |
| `result_dir` | 结果输出目录 | `result` |
| `programs_database.num_islands` | 岛屿数量 | 10 |
| `programs_database.reset_period` | 岛屿重置周期（秒） | 600 |
| `functional_dedup` | 功能级重复检测（Tier1 语法 + Tier2 执行输出） | True |
| `dedup_tier1_only` | 仅 Tier1 语法去重，不做 Tier2 | False |
| `progressive_eval` | 渐进式评估 | False |
| `adaptive_sampling` | 自适应采样 | False |
| `weighted_island_sampling` | 加权岛屿选择 | False |
| `feedback_in_prompt` | 失败反馈注入 prompt | False |

## 使用方法

### TSP 问题（随机实例）

```bash
python run.py --problem tsp
```

默认使用 3 个随机实例：`(15, 42)`, `(20, 123)`, `(25, 456)`（城市数, 随机种子）。

### TSP 问题（TSPLib 文件）

```bash
python run.py --problem tsp --tsplib path/to/att48.tsp path/to/eil51.tsp
```

### Cap Set / Admissible Set 问题

```bash
python run.py --problem admissible
```

### Sample-efficient 模式（减少评估与 LLM 调用）

```bash
# 启用全部样本效率优化
python run.py --problem tsp --tsplib data/tsplib/eil51.tsp \
  --progressive-eval --adaptive-sampling --weighted-island --feedback-in-prompt
```

功能级重复检测（`functional_dedup`）默认开启：Tier1 按代码 hash 去重，TSP 下 Tier2 按 tour 签名去重，避免对功能等价程序重复评估。

### 调试模式（显示 prompt 等 DEBUG 日志）

在运行前设置日志级别：

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

或在 `implementation/sampler.py` 中将 `logging.basicConfig(level=logging.INFO)` 改为 `logging.DEBUG`。

## 算法概述

### TSP 贪心构造

1. **框架**：每步用 `priority(city_idx, tour, unvisited, D)` 对候选城市打分，选分数最高的城市
2. **插入**：用最便宜插入法（cheapest insertion）将城市插入当前路径
3. **Baseline**：初始 priority 为最近邻（nearest-neighbor），LLM 可尝试最远插入、regret 等超越
4. **进化目标**：LLM 不断改进 `priority` 的实现，使 `evaluate` 返回的 `-tour_length` 越大越好

### FunSearch 流程

1. **初始化**：将规格中的初始 `priority` 注册到所有岛屿
2. **迭代**：每轮从某岛屿采样 prompt → LLM 生成新实现 → Evaluator 在测试实例上运行 → 合格则注册
3. **岛屿**：多岛屿保持多样性；定期重置弱岛屿，用强岛屿的最优程序作为 founder
4. **早停**：若连续 `early_stop_patience` 轮无全局改进，提前结束
5. **报告**：结束时生成 `result/YYYYMMDD_HHMMSS/` 下的图表与统计

## 输出说明

### result/YYYYMMDD_HHMMSS/

| 文件 | 内容 |
|------|------|
| `score_progression.png` | 全局最佳分数随迭代的变化曲线 |
| `final_results.json` | 总迭代数、token 消耗、最佳分数、分数历史等 |
| `best_program.py` | 当前最优的完整程序（含 `priority` 实现） |
| `experiment_summary.md` | 实验摘要（迭代数、token、最佳分数、运行时间等） |

### 日志级别

- **INFO**：每轮迭代摘要（tokens、分数、岛屿、重置次数等）、Cluster 统计
- **DEBUG**：发送给 LLM 的 prompt、evaluator 解析细节等

## 常见问题

### 1. `ValueError: zero-size array to reduction operation minimum`

LLM 生成的 `priority` 在边界情况（如只剩一个未访问城市）下对空数组调用 `np.min`。该样本会被视为无效，不注册到数据库；可忽略或通过改进 prompt 减少此类输出。

### 1b. `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'`

当 `priority` 返回 `None` 或非数值时，框架会将其视为 `-inf` 并跳过该候选，不再崩溃。

### 2. 早停未触发

若 `early_stop_patience` 大于 `iterations`，程序会在达到最大迭代数时正常结束，早停不会生效。可调小 `early_stop_patience` 或增大 `iterations`。

### 3. 导入错误

请确保在项目根目录运行 `python run.py`，以便 `implementation` 模块可被正确导入。

### 4. OpenRouter 404 / 权限错误

检查 API Key 是否正确，并在 OpenRouter 设置中放宽隐私与数据保留策略。

## 许可证

基于 DeepMind FunSearch 实现，遵循 Apache 2.0 许可证。
