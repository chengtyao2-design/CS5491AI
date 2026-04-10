# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a FunSearch experiment."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in prompts.
    num_islands: Number of islands to maintain as a diversity mechanism.
    reset_period: How often (in seconds) the weakest islands should be reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling
        of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster
        sampling temperature.
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = 10 * 60
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 50


@dataclasses.dataclass(frozen=True)
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the evolutionary algorithm.
    num_samplers: Number of independent Samplers in the experiment. A value
        larger than 1 only has an effect when the samplers are able to execute
        in parallel, e.g. on different matchines of a distributed system.
    num_evaluators: Number of independent program Evaluators in the experiment.
        A value larger than 1 is only expected to be useful when the Evaluators
        can execute in parallel as part of a distributed system.
    samples_per_prompt: How many independently sampled program continuations to
        obtain for each prompt.
    goal_description: Problem-specific goal text for LLM prompts (e.g. TSP vs
        admissible set).
    early_stop_patience: Number of iterations without improvement before early
        stopping. Set to -1 to disable.
    result_dir: Directory for saving experiment results (score plots, best
        program, etc.).
  """
  programs_database: ProgramsDatabaseConfig = dataclasses.field(
      default_factory=ProgramsDatabaseConfig)
  num_samplers: int = 1
  num_evaluators: int = 140
  samples_per_prompt: int = 4
  iterations: int = 50
  goal_description: str = (
      "maximize the size of the admissible set"
  )
  early_stop_patience: int = -1  # 早停的条件
  result_dir: str = "result"
  problem: str = "admissible"  # Problem name for result subdir: {problem}_{unix_timestamp}

  # Sample efficiency options（全关）
  progressive_eval: bool = False
  stage1_timeout: int = 5
  stage1_score_threshold_pct: float = 0.0  # 0 = 仅通过即可

  functional_dedup: bool = False
  dedup_tier1_only: bool = False  # 仅语法级去重，不做功能级检测

  adaptive_sampling: bool = False
  min_samples_per_prompt: int = 2
  max_samples_per_prompt: int = 4
  reduce_after_no_improve: int = 3

  weighted_island_sampling: bool = False
  island_sampling_temperature: float = 1.0  # softmax T for weighted island selection; higher = more uniform
  feedback_in_prompt: bool = False

  # Adaptive temperature: reheat cluster sampling when stuck in local optima
  adaptive_temperature: bool = False
  reheat_after_no_improve: int = 5        # iterations without improvement before reheat
  reheat_temperature_multiplier: float = 3.0  # multiply cluster init temperature by this factor when reheating
