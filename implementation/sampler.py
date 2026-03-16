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
      problem: str = "admissible",
      template: code_manipulation.Program | None = None,
      function_to_evolve: str = "",
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt, goal_description)
    self._max_iterations = max_iterations
    self._early_stop_patience = early_stop_patience
    self._result_dir = result_dir
    self._problem = problem
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._start_time = time.time()
    self._global_best_history: list[float] = []
    self._per_instance_score_history: list[dict[str, float]] = []
    self._per_instance_gap_history: list[dict[str, float]] = []
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
      
      # Record per-instance score and gap history (best so far each iteration)
      best_program_info = self._database.get_global_best_program()
      if best_program_info:
        best_island_id = best_program_info[1]
        scores_per_test = self._database._best_scores_per_test_per_island[best_island_id]
        if scores_per_test:
          score_dict: dict[str, float] = {}
          gap_dict: dict[str, float] = {}
          for instance, score in scores_per_test.items():
            instance_id = getattr(instance, "instance_id", str(instance))
            score_dict[instance_id] = float(score)
            optimal = getattr(instance, "optimal_tour_length", None)
            if optimal is not None and optimal > 0:
              tour_length = -float(score)
              gap_dict[instance_id] = (tour_length - optimal) / optimal * 100
          self._per_instance_score_history.append(score_dict)
          self._per_instance_gap_history.append(gap_dict)
        else:
          # No valid scores yet, carry forward previous or append empty
          prev_scores = self._per_instance_score_history[-1] if self._per_instance_score_history else {}
          prev_gaps = self._per_instance_gap_history[-1] if self._per_instance_gap_history else {}
          self._per_instance_score_history.append(prev_scores)
          self._per_instance_gap_history.append(prev_gaps)
      else:
        prev_scores = self._per_instance_score_history[-1] if self._per_instance_score_history else {}
        prev_gaps = self._per_instance_gap_history[-1] if self._per_instance_gap_history else {}
        self._per_instance_score_history.append(prev_scores)
        self._per_instance_gap_history.append(prev_gaps)
      
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
    
    unix_timestamp = int(time.time())
    out_dir = Path(self._result_dir) / f"{self._problem}_{unix_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save score progression plot (per-instance curves + average)
    try:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      from matplotlib.ticker import MaxNLocator
      
      plt.figure(figsize=(10, 6))
      iters = list(range(1, len(self._global_best_history) + 1))
      instance_ids = sorted(set(
        iid for h in self._per_instance_score_history for iid in h.keys()
      ))
      if instance_ids:
        for instance_id in instance_ids:
          scores = [
            h.get(instance_id) for h in self._per_instance_score_history
            if h.get(instance_id) is not None
          ]
          if scores:
            # Align length: use last known value for early iters without data
            full_scores = []
            last = None
            for h in self._per_instance_score_history:
              v = h.get(instance_id)
              last = v if v is not None else last
              full_scores.append(last)
            plt.plot(iters, full_scores, "-", linewidth=1, label=instance_id, alpha=0.8)
        avg_scores = [
          sum(h.values()) / len(h) if h else None
          for h in self._per_instance_score_history
        ]
        last_avg = None
        full_avg = []
        for v in avg_scores:
          last_avg = v if v is not None else last_avg
          full_avg.append(last_avg)
        if any(x is not None for x in full_avg):
          plt.plot(iters, full_avg, "k--", linewidth=2, label="Average")
        plt.legend(loc="best", fontsize=8)
      else:
        plt.plot(iters, self._global_best_history, "b-", linewidth=1, label="Global Best")
      plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
      plt.gca().set_xticks(iters)
      plt.xlabel("Iteration")
      plt.ylabel("Score")
      plt.title("Score Progression")
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      plt.savefig(out_dir / "score_progression.png", dpi=150)
      plt.close()
    except ImportError:
      logging.warning("matplotlib not installed, skipping score plot.")
    
    # 2. Build final_data and compute optimal comparison
    global_best = self._database.get_global_best_score()
    if global_best == -float("inf") and self._global_best_history:
        global_best = max(self._global_best_history)
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
    
    # Compute optimal comparison (gap vs known optimum) for TSPLib instances
    optimal_comparison: list[dict] = []
    if best_program_info:
      best_island_id = best_program_info[1]
      scores_per_test = self._database._best_scores_per_test_per_island[best_island_id]
      if scores_per_test:
        for instance, score in scores_per_test.items():
          optimal = getattr(instance, "optimal_tour_length", None)
          if optimal is not None and optimal > 0:
            tour_length = -float(score)
            gap_pct = (tour_length - optimal) / optimal * 100
            optimal_comparison.append({
                "instance_id": instance.instance_id,
                "tour_length": round(tour_length, 6),
                "optimal": optimal,
                "gap_pct": round(gap_pct, 4),
            })
    if optimal_comparison:
      final_data["optimal_comparison"] = optimal_comparison
      final_data["avg_gap_pct"] = round(
          sum(c["gap_pct"] for c in optimal_comparison) / len(optimal_comparison), 4
      )
    
    # Per-instance score and gap history (all evaluate scores)
    instance_ids = sorted(set(
      iid for h in self._per_instance_score_history for iid in h.keys()
    ))
    if instance_ids:
      final_data["score_history_per_instance"] = {
        iid: [
          round(h.get(iid), 6) if h.get(iid) is not None else None
          for h in self._per_instance_score_history
        ]
        for iid in instance_ids
      }
      avg_score_history = []
      for h in self._per_instance_score_history:
        if h:
          avg_score_history.append(round(sum(h.values()) / len(h), 6))
        else:
          avg_score_history.append(None)
      final_data["avg_score_history"] = avg_score_history
    if self._per_instance_gap_history and any(h for h in self._per_instance_gap_history):
      gap_instance_ids = sorted(set(
        iid for h in self._per_instance_gap_history for iid in h.keys()
      ))
      final_data["gap_history_per_instance"] = {
        iid: [
          round(h.get(iid), 4) if h.get(iid) is not None else None
          for h in self._per_instance_gap_history
        ]
        for iid in gap_instance_ids
      }
      avg_gap_history = []
      for h in self._per_instance_gap_history:
        if h:
          avg_gap_history.append(round(sum(h.values()) / len(h), 4))
        else:
          avg_gap_history.append(None)
      final_data["avg_gap_history"] = avg_gap_history
    
    # 2b. Save optimal comparison chart (when we have optimal_comparison data)
    if optimal_comparison:
      try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        ids = [c["instance_id"] for c in optimal_comparison]
        gaps = [c["gap_pct"] for c in optimal_comparison]
        avg_gap = final_data.get("avg_gap_pct", sum(gaps) / len(gaps))
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(ids)), gaps, color="steelblue", edgecolor="navy", alpha=0.8)
        plt.axhline(y=avg_gap, color="red", linestyle="--", linewidth=1, label=f"Avg: {avg_gap:.2f}%")
        plt.xticks(range(len(ids)), ids, rotation=45, ha="right")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Instance")
        plt.ylabel("Gap (%)")
        plt.title("Optimal Comparison: Gap vs Known Optimum")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "optimal_comparison.png", dpi=150)
        plt.close()
      except ImportError:
        logging.warning("matplotlib not installed, skipping optimal comparison plot.")
    
    # 2c. Save gap progression plot (per-instance curves + average)
    if self._per_instance_gap_history and any(h for h in self._per_instance_gap_history):
      try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        iters = list(range(1, len(self._per_instance_gap_history) + 1))
        instance_ids = sorted(set(
          iid for h in self._per_instance_gap_history for iid in h.keys()
        ))
        if instance_ids:
          plt.figure(figsize=(10, 6))
          for instance_id in instance_ids:
            full_gaps = []
            last = None
            for h in self._per_instance_gap_history:
              v = h.get(instance_id)
              last = v if v is not None else last
              full_gaps.append(last)
            if any(x is not None for x in full_gaps):
              plt.plot(iters, full_gaps, "-", linewidth=1, label=instance_id, alpha=0.8)
          avg_gaps = [
            sum(h.values()) / len(h) if h else None
            for h in self._per_instance_gap_history
          ]
          last_avg = None
          full_avg = []
          for v in avg_gaps:
            last_avg = v if v is not None else last_avg
            full_avg.append(last_avg)
          if any(x is not None for x in full_avg):
            plt.plot(iters, full_avg, "k--", linewidth=2, label="Average")
          plt.legend(loc="best", fontsize=8)
          plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
          plt.gca().set_xticks(iters)
          plt.xlabel("Iteration")
          plt.ylabel("Gap (%)")
          plt.title("Gap Progression vs Known Optimum")
          plt.grid(True, alpha=0.3)
          plt.tight_layout()
          plt.savefig(out_dir / "gap_progression.png", dpi=150)
          plt.close()
      except ImportError:
        logging.warning("matplotlib not installed, skipping gap progression plot.")
    
    with open(out_dir / "final_results.json", "w", encoding="utf-8") as f:
      json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # 3. Save best program code
    if best_program_info and best_program_info[0] is not None and self._template is not None and self._function_to_evolve:
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
      if optimal_comparison:
        f.write("\n## Optimal Comparison\n\n")
        f.write("| Instance | Tour Length | Optimal | Gap (%) |\n")
        f.write("|----------|-------------|---------|----------|\n")
        for c in optimal_comparison:
          f.write(f"| {c['instance_id']} | {c['tour_length']} | {c['optimal']} | {c['gap_pct']} |\n")
        f.write(f"\nAverage gap: {final_data['avg_gap_pct']}%\n\n")
        f.write("See `optimal_comparison.png` and `gap_progression.png` for the visualizations.\n")
    
    logging.info("Results saved to %s", out_dir)
    return out_dir
