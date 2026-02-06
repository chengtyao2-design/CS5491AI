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

"""The sampler class."""
import os
import time
import re
from collections.abc import Sequence

from absl import logging
import numpy as np
from openai import OpenAI

from implementation import evaluator
from implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""
  
  _total_tokens_used = 0

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

  def draw_samples(self, prompt: str) -> Sequence[str]:
    """Returns multiple predicted continuations of `prompt`."""
    model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
    user_prompt = (
        f"You are designing a heuristic for a Greedy Algorithm to build a maximal Constant Weight Admissible Set (Cap Set).\n\n"
        f"Please provide {self._samples_per_prompt} DISTINCT and improved implementations for the function body of `priority` in the following code.\n\n"
        f"CRITICAL REQUIREMENT: The {self._samples_per_prompt} implementations must be FUNDAMENTALLY DIFFERENT from each other and use different mathematical strategies/heuristics to optimize the set size from different angles. Do not just change constants.\n"
        f"Possible angles to explore: preferring specific values, pattern avoidance, symmetry breaking, randomness, or hybrid approaches.\n\n"
        f"The goal is to maximize the size of the admissible set. \n\n"
        f"Each implementation MUST be enclosed in a separate Python code block (```python ... ```).\n\n"
        f"{prompt}"
    )
    print(f"Prompt sent to LLM (requesting {self._samples_per_prompt} samples):\n---\n{prompt}\n---\n")
    
    retries = 5
    for attempt in range(retries):
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a mathematician and algorithm designer. Your goal is to design a scoring function `priority(el, n, w)` for a Greedy Algorithm that constructs a maximal **Constant Weight Admissible Set**.\n\n**Task Description:**\n- The Greedy Algorithm iteratively adds the candidate element `el` that maximizes your `priority` function.\n- Your goal is to design a heuristic that predicts which elements are 'best' to add early to allow the set to grow larger later.\n\n**Input Specs:**\n- `el`: A sparse vector of length `n` with weight `w` (entries are 0, 1, 2).\n\nSTRICTLY output only the function body code. Each implementation in a separate markdown code block. NO comments outside code blocks."},
                    {"role": "user", "content": user_prompt}
                ],
            )
            if resp.usage:
                LLM._total_tokens_used += resp.usage.total_tokens
            
            content = resp.choices[0].message.content
            
            # Parse multiple code blocks
            samples = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
            if not samples:
                samples = re.findall(r'```(.*?)```', content, re.DOTALL)
            
            clean_samples = [s.strip() for s in samples if s.strip()]
            
            # Fallback if no blocks found but content exists
            if not clean_samples and content.strip():
                 if 'return' in content:
                     clean_samples = [content.strip()]

            if clean_samples:
                return clean_samples
                
        except Exception as e:
            print(f"LLM API call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("Max retries reached. Stopping execution.")
                raise e
    return []
    
  @property
  def total_tokens_used(self) -> int:
      return LLM._total_tokens_used


class Sampler:
  """Node that samples programs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      max_iterations: int,
      log_file_path: str = None,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)
    self._start_time = time.time()
    self._max_iterations = max_iterations
    
    self._log_file_path = log_file_path
    self._last_global_best = -float('inf')

  def sample(self):
    """Continuously samples programs."""
    logging.info('Sampler started.')
    iteration = 0
    while iteration < self._max_iterations:
      prompt_code, version_generated, island_id = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt_code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, island_id, version_generated)
      
      iteration += 1
      
      best_score = self._database._best_score_per_island[island_id]
      
      # Calculate additional stats
      num_islands = len(self._database._islands)
      active_islands = sum(1 for score in self._database._best_score_per_island if score > -float('inf'))
      global_best_score = max(self._database._best_score_per_island)
      
      # Log to file if Global Best improved
      if global_best_score > self._last_global_best:
          self._last_global_best = global_best_score
          # Find the program with the global best score
          best_programs = self._database.get_best_programs_per_island()
          best_program_code = ""
          for program, score, _ in best_programs:
              if score == global_best_score and program:
                  best_program_code = str(program)
                  break
          
          log_entry = (f"{iteration} | {global_best_score} | {self._llm.total_tokens_used} | {self._database._reset_count} | "
                       f"Best Function:\n{best_program_code}\n"
                       f"{'-'*80}\n")
          
          with open(self._log_file_path, 'a') as f:
              f.write(log_entry)
          print(f"New Global Best! Logged to {self._log_file_path}")
      
      elapsed_seconds = int(time.time() - self._start_time)
      
      # Get current temperature of the island
      current_temp = self._database._islands[island_id].get_current_temperature()

      print(f"Iteration: {iteration} | Total Tokens: {self._llm.total_tokens_used} | "
            f"Best Score (Island {island_id}): {best_score} | "
            f"Global Best: {global_best_score} | "
            f"Active Islands: {active_islands}/{num_islands} | "
            f"Elapsed: {elapsed_seconds}s | "
            f"Resets: {self._database._reset_count} | "
            f"Duplicates: {self._database._duplicate_count}")

      print("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
          if not island._clusters:
              continue
          print(f"  Island {i} (Temp: {island.get_current_temperature():.1f}): {len(island._clusters)} clusters")
          for sig, cluster in island._clusters.items():
              print(f"    Cluster {sig} (Score: {cluster.score}): {len(cluster._programs)} programs")
