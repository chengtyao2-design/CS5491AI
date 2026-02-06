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

"""Class for sampling new programs."""
import os
from collections.abc import Collection, Sequence

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from implementation import evaluator
from implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    self.total_tokens_used = 0

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
    user_prompt = f"Please provide the implementation for the function body of `priority` in the following code. The goal is to maximize the size of the admissible set. \n\n{prompt}"
    print(f"Prompt sent to LLM:\n---\n{prompt}\n---\n")
    resp = self.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a search algorithm engineer. Your goal is to improve the computational logic of the function body. STRICTLY adhere to the function's signature and return type. DO NOT change the function's category or purpose. Write concise, high-performance code using branching structures or loops if necessary. Output code only, STRICTLY NO MARKDOWN and NO COMMENTS USING '#'. Your response should contain ONLY the function body code. Do not repeat the function signature or docstring."},
            {"role": "user", "content": user_prompt}
        ],
    )
    if resp.usage:
        self.total_tokens_used += resp.usage.total_tokens
    return resp.choices[0].message.content

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      max_iterations: int = -1,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)
    self._max_iterations = max_iterations

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    iteration = 0
    while self._max_iterations == -1 or iteration < self._max_iterations:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
      
      iteration += 1
      
      best_score = self._database._best_score_per_island[prompt.island_id]
      
      # Calculate additional stats
      num_islands = len(self._database._islands)
      active_islands = sum(1 for score in self._database._best_score_per_island if score > -float('inf'))
      global_best_score = max(self._database._best_score_per_island)
      
      print(f"Iteration: {iteration} | Total Tokens: {self._llm.total_tokens_used} | "
            f"Best Score (Island {prompt.island_id}): {best_score} | "
            f"Global Best: {global_best_score} | "
            f"Active Islands: {active_islands}/{num_islands}")

      print("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
          if not island._clusters:
              continue
          print(f"  Island {i}: {len(island._clusters)} clusters")
          for sig, cluster in island._clusters.items():
              print(f"    Cluster {sig} (Score: {cluster.score}): {len(cluster._programs)} programs")
