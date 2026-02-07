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
        timeout=30.0,
    )

  def _parse_samples(self, content: str) -> tuple[Sequence[str], int]:
    """Parses code blocks from LLM response."""
    # Parse multiple code blocks
    samples = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not samples:
        samples = re.findall(r'```(.*?)```', content, re.DOTALL)
    
    clean_samples = [s.strip() for s in samples if s.strip()]
    
    # Robust Deduplication
    unique_samples = []
    seen_fingerprints = set()
    for sample in clean_samples:
        # 1. Remove docstrings
        content_no_doc = re.sub(r'""".*?"""', '', sample, flags=re.DOTALL)
        content_no_doc = re.sub(r"'''.*?'''", '', content_no_doc, flags=re.DOTALL)
        
        # 2. Remove function headers/decorators to compare body only
        lines = [line for line in content_no_doc.splitlines() 
                 if not line.strip().startswith(('def ', '@'))]
        body_content = '\n'.join(lines)

        # 3. Normalize whitespace (collapse all whitespace to single space)
        fingerprint = re.sub(r'\s+', ' ', body_content).strip()
        
        if fingerprint and fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_samples.append(sample)
    
    duplicate_count = len(clean_samples) - len(unique_samples)
    clean_samples = unique_samples
    
    # Fallback if no blocks found but content exists
    if not clean_samples and content.strip():
         if 'return' in content:
             clean_samples = [content.strip()]

    return clean_samples, duplicate_count

  def resolve_error(self, prompt_code: str, bad_code: str, error_msg: str) -> str | None:
      """Attempts to fix the code based on the error message."""
      model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
      temperature = 0.4 # Lower temp for fixing
      
      system_prompt = "You are a mathematician and algorithm designer. Your goal is to design a scoring function `priority(el, n, w)` for a Greedy Algorithm that constructs a maximal **Constant Weight Admissible Set**.\n\nSTRICTLY output only the function body code. Each implementation in a separate markdown code block. NO comments outside code blocks."
      
      # We construct a simplified user prompt to remind the model of the task.
      user_prompt = (
        f"You are designing a heuristic for a Greedy Algorithm.\n"
        f"Please fix the following implementation of `priority` which failed with an error.\n\n"
        f"Context Code:\n{prompt_code}\n\n"
      )
      
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
          {"role": "assistant", "content": f"```python\n{bad_code}\n```"},
          {"role": "user", "content": f"The code above failed with the following error:\n{error_msg}\n\nPlease fix the code to resolve this error. STRICTLY output only the corrected function body code in a python code block."}
      ]
      
      print(f"DEBUG: Requesting fix for error...")

      retries = 2
      for attempt in range(retries):
          try:
              resp = self.client.chat.completions.create(
                  model=model,
                  temperature=temperature,
                  messages=messages,
              )
              if resp.usage:
                  LLM._total_tokens_used += resp.usage.total_tokens
              
              content = resp.choices[0].message.content
              samples, _ = self._parse_samples(content)
              
              if samples:
                  return samples[0]
          except Exception as e:
              print(f"LLM fix call failed: {e}")
              time.sleep(1)
      return None

  def draw_samples(self, prompt: str, temperature: float) -> tuple[Sequence[str], int]:
    """Returns multiple predicted continuations of `prompt` and the duplicate count."""
    model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
    
    # Conditional Constraint: Only added when temperature is high (system is stuck/repetitive)
    force_change_prompt = ""
    if temperature >= 0.9:
        force_change_prompt = (
            f"Constraint 4 (High Temperature Override): You are currently stuck in a local optimum. "
            f"You MUST drastically change the scoring logic. Do NOT use the same `if/else` structure as the previous version. "
            f"Invent a completely NEW formula for the priority score.\n"
        )

    user_prompt = (
        f"You are designing a heuristic for a Greedy Algorithm to build a maximal Constant Weight Admissible Set (Cap Set).\n\n"
        f"Please provide {self._samples_per_prompt} DISTINCT and improved implementations for the function body of `priority` in the following code.\n\n"
        f"CRITICAL REQUIREMENT: The {self._samples_per_prompt} implementations must be FUNDAMENTALLY DIFFERENT from each other and use different mathematical strategies/heuristics to optimize the set size from different angles. Do not just change constants.\n"
        f"Possible angles to explore: preferring specific values, pattern avoidance, symmetry breaking, randomness, or hybrid approaches.\n\n"
        f"IMPORTANT: The functions in the prompt are versioned (e.g., `_v0`, `_v1`, ...). Higher version numbers (larger `k` in `_vk`) generally indicate better performance. You should pay SPECIAL ATTENTION to the logic of the latest version and try to improve upon it. \n"
        f"Constraint 1: The {self._samples_per_prompt} improved versions you generate MUST use completely different algorithmic logic from each other. They should be diverse approaches to solving the problem, not variations of the same idea.\n"
        f"Constraint 2: Do NOT generate a sequence of incremental improvements (v2 -> v3 -> v4). Instead, generate {self._samples_per_prompt} INDEPENDENT alternatives. Each alternative must be derived directly from the prompt's context, ignoring other alternatives generated in this response.\n"
        f"Constraint 3: Do NOT use version numbers in function names (e.g., NO `priority_v2`, `priority_v3`). Use distinct names or just `priority`.\n"
        f"{force_change_prompt}"
        f"Each implementation MUST be enclosed in a separate Python code block (```python ... ```).\n\n"
        f"{prompt}"
    )
    print(f"Prompt sent to LLM (requesting {self._samples_per_prompt} samples, Temp: {temperature:.2f}):\n---\n{prompt}\n---\n")
    
    retries = 5
    for attempt in range(retries):
        try:
            resp = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a mathematician and algorithm designer. Your goal is to design a scoring function `priority(el, n, w)` for a Greedy Algorithm that constructs a maximal **Constant Weight Admissible Set**.\n\n**Task Description:**\n- The Greedy Algorithm iteratively adds the candidate element `el` that maximizes your `priority` function.\n- Your goal is to design a heuristic that predicts which elements are 'best' to add early to allow the set to grow larger later.\n\n**Input Specs:**\n- `el`: A sparse vector of length `n` with weight `w` (entries are 0, 1, 2).\n\nSTRICTLY output only the function body code. Each implementation in a separate markdown code block. NO comments outside code blocks."},
                    {"role": "user", "content": user_prompt}
                ],
            )
            if resp.usage:
                LLM._total_tokens_used += resp.usage.total_tokens
            
            content = resp.choices[0].message.content
            
            clean_samples, duplicate_count = self._parse_samples(content)

            if clean_samples:
                return clean_samples, duplicate_count
                
        except Exception as e:
            print(f"LLM API call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("Max retries reached. Stopping execution.")
                raise e
    return [], 0
    
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
    self._llm_temperature = 0.6
    self._epoch = 1

  def sample(self):
    """Continuously samples programs."""
    logging.info('Sampler started.')
    iteration = 0
    while True: # Run forever (until manual stop or other condition)
      # Reset iteration counter and increment epoch if max_iterations reached
      if iteration >= self._max_iterations:
          self._epoch += 1
          iteration = 0
          print(f"\n{'='*40}\nStarting Epoch {self._epoch}\n{'='*40}\n")
      
      prompt_code, version_generated, island_id = self._database.get_prompt(self._epoch)
      samples, num_duplicates = self._llm.draw_samples(prompt_code, self._llm_temperature)
      
      # Dynamic Temperature Adjustment
      if num_duplicates > 0:
          # If we see duplicates, increase temperature to encourage diversity
          self._llm_temperature = min(1.0, self._llm_temperature + 0.05)
      else:
          # If no duplicates, slowly cool down to exploit
          self._llm_temperature = max(0.6, self._llm_temperature - 0.01)

      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        error_msg = chosen_evaluator.analyse(
            sample, island_id, version_generated)
        
        if error_msg:
             # Fix loop
             current_code = sample
             current_error = error_msg
             
             # Reduce fix attempts to 1 to prevent long stalls
             for fix_attempt in range(1):
                 print(f"DEBUG: Attempting to fix code (Attempt {fix_attempt+1}/1).")
                 
                 fixed_code = self._llm.resolve_error(prompt_code, current_code, current_error)
                 
                 if not fixed_code:
                     print("DEBUG: LLM failed to provide fix.")
                     break
                 
                 # Eval fixed code
                 current_error = chosen_evaluator.analyse(fixed_code, island_id, version_generated)
                 
                 if not current_error:
                     print("DEBUG: Fix successful! Code registered.")
                     break
                 
                 current_code = fixed_code
                 print(f"DEBUG: Fix failed again.")
      
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
          best_scores_str = str(global_best_score)
          
          for program, score, scores_per_test in best_programs:
              if score == global_best_score and program:
                  best_program_code = str(program)
                  if scores_per_test:
                      # Format as (val1, val2, ...) (Score: total) to show details for multi-input tasks
                      # Sort keys to ensure deterministic order of values
                      sorted_vals = [scores_per_test[k] for k in sorted(scores_per_test.keys())]
                      vals_str = ", ".join(str(v) for v in sorted_vals)
                      best_scores_str = f"({vals_str}) (Score: {score})"
                  break
          
          # Calculate current dynamic parameters for logging
          current_funcs_per_prompt = self._database._functions_per_prompt
          if self._epoch > 3:
              current_funcs_per_prompt += (self._epoch - 3)
          
          # Replace strategy logic (simplified check based on epoch and cluster count approximation)
          # Note: Actual replace strategy depends on specific island state, here we log the Policy Intent.
          replace_strategy = "True"
          if self._epoch >= 2:
               replace_strategy = "False (Intent)"

          log_entry = (f"{self._epoch} | {iteration} | {best_scores_str} | {current_funcs_per_prompt} | {replace_strategy} | "
                       f"{self._llm.total_tokens_used} | {self._database._reset_count} | "
                       f"Best Function:\n{best_program_code}\n"
                       f"{'-'*80}\n")
          
          with open(self._log_file_path, 'a') as f:
              f.write(log_entry)
          print(f"New Global Best! Logged to {self._log_file_path}")
      
      elapsed_seconds = int(time.time() - self._start_time)
      
      # Get current temperature of the island
      current_temp = self._database._islands[island_id].get_current_temperature()

      print(f"Epoch: {self._epoch} | Iteration: {iteration} | Total Tokens: {self._llm.total_tokens_used} | "
            f"Best Score (Island {island_id}): {best_score} | "
            f"Global Best: {global_best_score} | "
            f"Active Islands: {active_islands}/{num_islands} | "
            f"Elapsed: {elapsed_seconds}s | "
            f"Resets: {self._database._reset_count} | "
            f"LLM Temp: {self._llm_temperature:.2f} | "
            f"LLM Dups: {num_duplicates} | "
            f"DB Dups: {self._database._duplicate_count}")

      print("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
          if not island._clusters:
              continue
          
          # Check if pruning is active (num_programs > 12)
          pruning_status = " [PRUNING]" if island._num_programs > 12 else ""
          
          print(f"  Island {i} (Temp: {island.get_current_temperature():.1f}){pruning_status}: {len(island._clusters)} clusters")
          for sig, cluster in island._clusters.items():
              print(f"    Cluster {sig} (Score: {cluster.score}): {len(cluster._programs)} programs")
