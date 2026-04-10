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

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]

# #region agent log
def _agent_dbg(msg: str, hypothesis_id: str, data: dict) -> None:
  import json as _json
  try:
    with open(
        "/Users/yangkefan/Documents/projects/CS5491AI/.cursor/debug-60277f.log",
        "a",
        encoding="utf-8",
    ) as _f:
      _f.write(
          _json.dumps(
              {
                  "sessionId": "60277f",
                  "hypothesisId": hypothesis_id,
                  "location": "programs_database",
                  "message": msg,
                  "data": data,
                  "timestamp": int(time.time() * 1000),
              },
              ensure_ascii=False,
          )
          + "\n"
      )
  except Exception:
    pass
# #endregion


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  # #region agent log
  _agent_dbg(
      "_softmax entry",
      "H1",
      {
          "logits_shape": [int(x) for x in np.shape(logits)],
          "logits_size": int(np.size(logits)),
          "temperature": float(temperature),
      },
  )
  # #endregion
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  # return scores_per_test[list(scores_per_test.keys())[-1]]
  return sum(scores_per_test.values()) / len(scores_per_test)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
      config_full: Any = None,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._config_full: Any = config_full

    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    self._last_reset_time: float = time.time()
    self._reset_count: int = 0
    self._recent_failures: list[tuple[str, str]] = []
    self._no_improve_count: int = 0

    # Efficiency stats for sample-efficient reporting
    self._eff_samples_attempted: int = 0
    self._eff_full_evaluations: int = 0
    self._eff_skipped: dict[str, int] = {}

  def record_sample_attempted(self) -> None:
    """Record that a sample was sent to the evaluator."""
    self._eff_samples_attempted += 1

  def record_skip(self, reason: str) -> None:
    """Record that a sample was skipped (early return) for the given reason."""
    self._eff_skipped[reason] = self._eff_skipped.get(reason, 0) + 1

  def record_full_eval(self) -> None:
    """Record that a sample completed full evaluation and was registered."""
    self._eff_full_evaluations += 1

  def get_efficiency_stats(self) -> dict[str, Any]:
    """Return a read-only copy of efficiency statistics."""
    return {
        "samples_attempted": self._eff_samples_attempted,
        "full_evaluations": self._eff_full_evaluations,
        "skipped": dict(self._eff_skipped),
    }

  def notify_no_improve(self, count: int) -> None:
    """Called by Sampler each iteration to update the no-improvement counter."""
    self._no_improve_count = count

  def record_failure(self, code_snippet: str, reason: str) -> None:
    """Record a rejected sample for feedback-in-prompt."""
    flat = code_snippet.replace('\n', ' ').strip()
    snippet = (flat[:200] + "...") if len(flat) > 200 else flat
    self._recent_failures.append((snippet, reason))
    if len(self._recent_failures) > 2:
      self._recent_failures.pop(0)

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    if (
        self._config_full is not None
        and getattr(self._config_full, 'weighted_island_sampling', False)
    ):
      scores = np.array(self._best_score_per_island, dtype=np.float32)
      scores = np.where(np.isfinite(scores), scores, -1e9)
      island_temp = float(getattr(self._config_full, 'island_sampling_temperature', 1.0))
      probs = _softmax(scores, island_temp)
      island_id = int(np.random.choice(len(self._islands), p=probs))
    else:
      island_id = np.random.randint(len(self._islands))
    # Compute adaptive temperature multiplier for cluster sampling
    temp_multiplier = 1.0
    if (
        self._config_full is not None
        and getattr(self._config_full, 'adaptive_temperature', False)
    ):
      reheat_threshold = int(getattr(self._config_full, 'reheat_after_no_improve', 5))
      reheat_mult = float(getattr(self._config_full, 'reheat_temperature_multiplier', 3.0))
      if self._no_improve_count >= reheat_threshold:
        temp_multiplier = reheat_mult

    code, version_generated = self._islands[island_id].get_prompt(temp_multiplier)
    if (
        self._config_full is not None
        and getattr(self._config_full, 'feedback_in_prompt', False)
        and self._recent_failures
    ):
      feedback_lines = []
      for snippet, reason in self._recent_failures[-2:]:
        short = (snippet[:100] + "...") if len(snippet) > 100 else snippet
        feedback_lines.append(f"# Rejected (reason: {reason}):\n# {short}")
      code = code + "\n\n" + "\n".join(feedback_lines)
    return Prompt(code, version_generated, island_id)

  def get_global_best_score(self) -> float:
    """Returns the current global best score across all islands."""
    return max(self._best_score_per_island)

  def get_global_best_program(self) -> tuple[code_manipulation.Function, int] | None:
    """Returns (best_program, island_id) or None if no valid program yet."""
    best_score = self.get_global_best_score()
    if best_score == -float('inf'):
      return None
    for island_id, score in enumerate(self._best_score_per_island):
      if score == best_score and self._best_program_per_island[island_id] is not None:
        return self._best_program_per_island[island_id], island_id
    return None

  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the specified island."""
    self._islands[island_id].register_program(program, scores_per_test)
    score = _reduce_score(scores_per_test)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      self._best_score_per_island[island_id] = score
      logging.info('Best score of island %d increased to %s', island_id, score)

  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    self._reset_count += 1
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      # #region agent log
      _agent_dbg(
          "reset_islands founder",
          "H5",
          {
              "reset_island_id": island_id,
              "founder_island_id": int(founder_island_id),
              "founder_is_none": founder is None,
              "founder_scores_is_none": founder_scores is None,
          },
      )
      # #endregion
      self._register_program_in_island(founder, island_id, founder_scores)


class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0

  def register_program(
      self,
      program: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs += 1

  def get_prompt(self, temperature_multiplier: float = 1.0) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # #region agent log
    _agent_dbg(
        "Island.get_prompt pre-softmax",
        "H1",
        {
            "n_clusters": len(self._clusters),
            "cluster_scores_size": int(np.size(cluster_scores)),
            "num_programs": self._num_programs,
        },
    )
    # #endregion

    # Convert scores to probabilities using softmax with temperature schedule.
    # temperature_multiplier > 1 reheats (more exploration) when stuck.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period) * temperature_multiplier
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt) #**

    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self._function_to_evolve}_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    # Use higher temperature (e.g. 4.0) to reduce length penalty
    probabilities = _softmax(-normalized_lengths, temperature=4.0)
    return np.random.choice(self._programs, p=probabilities)
