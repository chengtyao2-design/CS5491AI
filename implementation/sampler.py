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
import copy
from implementation.programs_database import Island
import json
import logging
import os
import time
from collections.abc import Collection, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from implementation import code_manipulation
from implementation import evaluator
from implementation import programs_database

# Configure logging: default INFO, set to DEBUG to see prompts
logging.basicConfig(level=logging.INFO)


class LLM:
  """Language model that predicts continuation of provided source code."""

  _total_tokens_used = 0  # Class variable to persist across instances

  def __init__(
      self,
      samples_per_prompt: int,
      goal_description: str = "maximize the size of the admissible set",
  ) -> None:
    self._samples_per_prompt = samples_per_prompt
    self._goal_description = goal_description
    self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

  @property
  def total_tokens_used(self):
      return LLM._total_tokens_used

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
    user_prompt = (
        f"Please provide the implementation for the function body of `priority` "
        f"in the following code. The goal is to {self._goal_description}.\n\n{prompt}"
    )
    logging.debug("Prompt sent to LLM:\n---\n%s\n---", user_prompt)
    
    retries = 5
    for attempt in range(retries):
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a search algorithm engineer. Your goal is to improve the computational logic of the function body. STRICTLY adhere to the function's signature and return type. DO NOT change the function's category or purpose. Write concise, high-performance code using branching structures or loops if necessary. Output code only, STRICTLY NO MARKDOWN and NO COMMENTS USING '#'. Your response should contain ONLY the function body code. Do not repeat the function signature or docstring."},
                    {"role": "user", "content": user_prompt}
                ],
            )
            if resp.usage:
                LLM._total_tokens_used += resp.usage.total_tokens
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("Max retries reached. Stopping execution.")
                raise e

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
      goal_description: str = "maximize the size of the admissible set",
      early_stop_patience: int = -1,
      result_dir: str = "result",
      template: code_manipulation.Program | None = None,
      function_to_evolve: str = "",
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt, goal_description)
    self._max_iterations = max_iterations
    self._early_stop_patience = early_stop_patience
    self._result_dir = result_dir
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._start_time = time.time()
    self._global_best_history: list[float] = []
    self._last_improvement_iter = 0
    self._total_iterations = 0

  def sample(self) -> None:
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
      self._total_iterations = iteration
      
      best_score = self._database._best_score_per_island[prompt.island_id]
      global_best_score = self._database.get_global_best_score()
      
      # Track improvement for early stopping
      if not self._global_best_history or global_best_score > max(self._global_best_history):
        self._last_improvement_iter = iteration
      self._global_best_history.append(global_best_score)
      
      # Calculate additional stats
      num_islands = len(self._database._islands)
      active_islands = sum(1 for score in self._database._best_score_per_island if score > -float('inf'))
      elapsed_seconds = int(time.time() - self._start_time)
      
      # Check if this iteration found a new global best
      new_global_best = (
          len(self._global_best_history) <= 1 or
          global_best_score > self._global_best_history[-2]
      )

      logging.info(
          "Iteration: %d | Total Tokens: %d | Best Score (Island %d): %s | "
          "Global Best: %s%s | Active Islands: %d/%d | Elapsed: %ds | Resets: %d | "
          "Version: %d",
          iteration, self._llm.total_tokens_used, prompt.island_id, best_score,
          global_best_score, " (NEW!)" if new_global_best else "",
          active_islands, num_islands, elapsed_seconds,
          self._database._reset_count, prompt.version_generated,
      )
      
      # Cluster stats
      logging.info("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
        if not island._clusters:
          logging.info("  Island %d: 0 clusters", i)
          continue
        logging.info("  Island %d: %d clusters", i, len(island._clusters))
        for sig, cluster in island._clusters.items():
          logging.info("    Cluster %s (Score: %s): %d programs", sig, cluster.score, len(cluster._programs))
      
      # Early stopping check
      if self._early_stop_patience > 0 and iteration - self._last_improvement_iter >= self._early_stop_patience:
        logging.info(
            "Early stopping: No improvement for %d iterations (since iter %d)",
            self._early_stop_patience, self._last_improvement_iter,
        )
        break

  def generate_summary_report(self) -> Path | None:
    """Generates and saves experiment summary to result_dir with timestamp."""
    if not self._global_best_history:
      logging.warning("No iteration data to report.")
      return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(self._result_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save score progression plot
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      from matplotlib.ticker import MaxNLocator
      
      plt.figure(figsize=(10, 6))
      plt.plot(range(1, len(self._global_best_history) + 1), self._global_best_history, "b-", linewidth=1)
      plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
      plt.xlabel("Iteration")
      plt.ylabel("Global Best Score")
      plt.title("Score Progression")
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      plt.savefig(out_dir / "score_progression.png", dpi=150)
      plt.close()
    except ImportError:
      logging.warning("matplotlib not installed, skipping score plot.")
    
    # 2. Save final_results.json
    global_best = self._database.get_global_best_score()
    best_program_info = self._database.get_global_best_program()
    
    final_data = {
        "total_iterations": self._total_iterations,
        "total_tokens": self._llm.total_tokens_used,
        "global_best_score": global_best,
        "elapsed_seconds": int(time.time() - self._start_time),
        "score_history": self._global_best_history,
    }
    if best_program_info:
      final_data["best_island_id"] = best_program_info[1]
    
    with open(out_dir / "final_results.json", "w", encoding="utf-8") as f:
      json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # 3. Save best program code
    if best_program_info and self._template is not None and self._function_to_evolve:
      best_func, _ = best_program_info
      program = copy.deepcopy(self._template)
      evolved = program.get_function(self._function_to_evolve)
      evolved.body = best_func.body
      with open(out_dir / "best_program.py", "w", encoding="utf-8") as f:
        f.write(str(program))
    
    # 4. Save experiment summary markdown
    with open(out_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
      f.write("# Experiment Summary\n\n")
      f.write(f"- **Total Iterations**: {self._total_iterations}\n")
      f.write(f"- **Total Tokens**: {self._llm.total_tokens_used}\n")
      f.write(f"- **Global Best Score**: {global_best}\n")
      f.write(f"- **Elapsed Time**: {final_data['elapsed_seconds']}s\n")
      if best_program_info:
        f.write(f"- **Best Island**: {best_program_info[1]}\n")
      f.write("\n## Score Progression\n\n")
      f.write("See `score_progression.png` for the visualization.\n")
    
    logging.info("Results saved to %s", out_dir)
    return out_dir
