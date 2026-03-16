# TSPLib 标准测试实例

本目录包含 TSP 标准测试文件，用于 FunSearch 求解器 benchmark。支持 EUC_2D、ATT、GEO、EXPLICIT 等格式（通过 tsplib95）。

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

## 参数说明

- `--tsplib`：TSPLib 文件路径，可多个。指定后仅用 tsplib，忽略 `--random` / `--seed`
- `--random`：随机实例的城市数，可多个（如 `15 20 25`）
- `--seed`：随机实例的 seed，可选；若提供则数量须与 `--random` 一致，否则报错；未提供时默认 seed=0

## Gap 计算说明

- 含 **OPTIMUM** 或 **BEST_KNOWN** 的 `.tsp` 可直接计算 gap
- 仅有 `.opt.tour` 的实例需解析 tour 并计算路径长度得到 optimal；当前实现未支持，需在 `.tsp` 中手动添加 OPTIMUM 行

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
