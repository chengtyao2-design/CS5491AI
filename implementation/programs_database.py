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
import ast
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


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
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


def _get_program_fingerprint(program: code_manipulation.Function) -> str:
  try:
    return ast.dump(ast.parse(program.body), include_attributes=False)
  except Exception:
    return "".join(program.body.split())


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve

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

    self._functions_per_prompt: int = config.functions_per_prompt
    self._cluster_sampling_temperature_init: float = (
        config.cluster_sampling_temperature_init)
    self._cluster_sampling_temperature_period: int = (
        config.cluster_sampling_temperature_period)

    self._backup_islands: list[Island] | None = None
    self._last_reset_time: float = time.time()
    self._reset_count: int = 0
    # Global set to track all programs ever seen across all islands/resets
    self._seen_fingerprints: set[str] = set()
    self._duplicate_count: int = 0

  def get_best_programs_per_island(
      self) -> list[tuple[code_manipulation.Function | None, float,
                          ScoresPerTest | None]]:
    return list(
        zip(self._best_program_per_island, self._best_score_per_island,
            self._best_scores_per_test_per_island))

  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # Global deduplication: check if we've ever seen this code before
    fingerprint = _get_program_fingerprint(program)
    if fingerprint in self._seen_fingerprints:
      self._duplicate_count += 1
      logging.info(f"Duplicate program detected. Total duplicates: {self._duplicate_count}")
      return
    self._seen_fingerprints.add(fingerprint)

    if island_id is None:
      # This is the initial program, so we register it in all islands.
      for i in range(len(self._islands)):
        self._register_program_internal(program, i, scores_per_test)
    else:
      self._register_program_internal(program, island_id, scores_per_test)

    # Reset islands if necessary.
    if time.time() - self._last_reset_time > self._config.reset_period:
      self._last_reset_time = time.time()
      self.reset_islands()

  def _register_program_internal(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    # Record best score per island.
    score = _reduce_score(scores_per_test)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_score_per_island[island_id] = score
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      logging.info('Best score of island %d increased to %s', island_id, score)

    # Register program in island.
    self._islands[island_id].register_program(program, scores_per_test)

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    self._reset_count += 1
    # Sort islands by their best score.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

    # For each island to reset, pick a random island from the top half and
    # copy its content.
    for island_id in reset_islands_ids:
      founder_island_id = np.random.choice(keep_islands_ids)
      founder_island = self._islands[founder_island_id]
      founder_best_program = self._best_program_per_island[founder_island_id]
      founder_best_score = self._best_score_per_island[founder_island_id]
      founder_best_scores_per_test = self._best_scores_per_test_per_island[
          founder_island_id]
      self._islands[island_id] = copy.deepcopy(founder_island)
      self._best_program_per_island[island_id] = founder_best_program
      self._best_score_per_island[island_id] = founder_best_score
      self._best_scores_per_test_per_island[
          island_id] = founder_best_scores_per_test

  def get_current_temperature(self) -> float:
      """Returns the temperature used in the last sampling."""
      return getattr(self, '_current_temperature', self._cluster_sampling_temperature_init)

  def get_prompt(self, epoch: int = 1) -> tuple[str, int]:
    """Returns a prompt for the sampler."""
    # Pick a random island.
    island_id = np.random.randint(len(self._islands))
    prompt, version_generated = self._islands[island_id].get_prompt(epoch)
    return prompt, version_generated, island_id


class Island:
  """A sub-population of programs."""

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
    self._cluster_sampling_temperature_init: float = (
        cluster_sampling_temperature_init)
    self._cluster_sampling_temperature_period: int = (
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

    # Pruning: If total programs on this island exceed threshold, trigger evolutionary culling
    if self._num_programs > 12:
        programs_to_remove = 0
        
        # Zone 1 (13-16): Growth Phase (Add N, Delete 1)
        if self._num_programs <= 16:
            if np.random.random() < 0.25:
                programs_to_remove = 1
                
        # Zone 2 (>16): Overpopulation Phase (Add 1, Delete 2)
        else:
            programs_to_remove = 2
            
        for _ in range(programs_to_remove):
            if not self._clusters:
                break
                
            # Sort clusters by score (ascending)
            sorted_signatures = sorted(self._clusters.keys(), key=lambda sig: self._clusters[sig].score)
            
            # Soft Pruning: Randomly select from the bottom 2 clusters to remove FROM.
            # This prevents instant extinction of new, exploring clusters.
            candidates_count = min(len(sorted_signatures), 2)
            worst_sig = sorted_signatures[np.random.randint(candidates_count)]
            worst_cluster = self._clusters[worst_sig]
            
            # If there are ties for the worst score (only if we picked the absolute worst)
            if candidates_count == 1: # Only check ties if we are forced to pick the single worst
                min_score = worst_cluster.score
                tied_signatures = [sig for sig in sorted_signatures if self._clusters[sig].score == min_score]
                
                if len(tied_signatures) > 1:
                    worst_sig = tied_signatures[np.random.randint(len(tied_signatures))]
                    worst_cluster = self._clusters[worst_sig]        
            # ACTION: Remove ONE program from this cluster (not the whole cluster)
            worst_cluster.pop_random_program()
            self._num_programs -= 1
            
            # Only remove the cluster if it becomes empty
            if not worst_cluster._programs:
                del self._clusters[worst_sig]

  def get_current_temperature(self) -> float:
      """Returns the temperature used in the last sampling."""
      return getattr(self, '_current_temperature', self._cluster_sampling_temperature_init)

  def get_prompt(self, epoch: int = 1) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period)

    # Dynamic temperature adjustment:
    self._current_temperature = temperature # Store base temperature
    
    cluster_sizes = [len(c._programs) for c in self._clusters.values()]
    if len(cluster_sizes) > 1:
      max_ratio = max(cluster_sizes) / sum(cluster_sizes)
      min_possible_ratio = 1.0 / len(cluster_sizes)
      
      # Normalize concentration to [0, 1]
      # 0 = Perfectly uniform (max_ratio == 1/N)
      # 1 = Perfectly concentrated (max_ratio == 1.0)
      concentration = (max_ratio - min_possible_ratio) / (1.0 - min_possible_ratio + 1e-6)
      
      # Use power function to keep temp low when concentration is low
      # Temp = Base + (Max - Base) * concentration
      dynamic_temp = self._cluster_sampling_temperature_init + (50.0 - self._cluster_sampling_temperature_init) * concentration
      
      if dynamic_temp > temperature:
          temperature = dynamic_temp
          if concentration > 0.5: # Log only significant boosts (conc > 0.5)
             logging.info(f'Cluster concentration {concentration:.1%}. Boosting temperature to {temperature:.1f}')

    self._current_temperature = temperature # Update with boosted temperature
    probabilities = _softmax(cluster_scores, temperature)

    # Calculate dynamic functions_per_prompt based on epoch
    # If epoch > 5, increase prompt size by (epoch - 5)
    current_functions_per_prompt = self._functions_per_prompt
    if epoch > 5:
        current_functions_per_prompt += (epoch - 5)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), current_functions_per_prompt) #**

    # Dynamic Sampling Strategy:
    # 1. Default: If we have enough clusters, sample without replacement.
    replace = len(signatures) < functions_per_prompt
    
    # 2. Epoch >= 3 Rule: Force sampling without replacement if possible.
    #    AND sort indices by score to prioritize better programs if needed?
    #    Actually, np.random.choice with probabilities already handles prioritization.
    #    But the user asked for "sort by score" explicitly for epoch >= 3?
    #    The original code sorts the *selected* implementations by score later.
    #    Here we just select indices.
    
    if epoch >= 3:
        # User requested: "sample without replacement" (already handled by logic above if possible)
        # But if we force replace=False when len < needed, it crashes. So we keep safety check.
        replace = False if len(signatures) >= functions_per_prompt else True
        
    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities, replace=replace)
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
    self._program_fingerprints: set[str] = {
        _get_program_fingerprint(implementation)
    }

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    fingerprint = _get_program_fingerprint(program)
    if fingerprint in self._program_fingerprints:
      return  # Skip duplicate

    self._programs.append(program)
    self._lengths.append(len(str(program)))
    self._program_fingerprints.add(fingerprint)

  def pop_random_program(self) -> None:
    """Removes a random program from the cluster."""
    if not self._programs:
        return
    idx = np.random.randint(len(self._programs))
    program = self._programs.pop(idx)
    self._lengths.pop(idx)
    # Note: We don't remove from _program_fingerprints to allow re-discovery
    # and to keep operations simple/fast.

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    # Use higher temperature (e.g. 4.0) to reduce length penalty
    probabilities = _softmax(-normalized_lengths, temperature=4.0)
    return np.random.choice(self._programs, p=probabilities)
