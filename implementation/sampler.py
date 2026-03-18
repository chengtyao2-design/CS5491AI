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
import time
from collections.abc import Collection, Sequence

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from implementation import evaluator
from implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  _total_tokens_used = 0  # Class variable to persist across instances

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
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
    user_prompt = f"Please provide the implementation for the function body of `priority` in the following code. The goal is to maximize the size of the admissible set.\n\nHINT: Think outside the box. Consider penalizing elements that block too many future valid combinations. **Crucially, add a small random noise or use a hashing mechanism to break ties between elements that evaluate to the same basic score. This prevents the search from getting stuck.**\n\nFirst, inside <Thought> tags, briefly analyze the previous code and explain your new idea. Then, inside <Code> tags, provide ONLY the Python code for the function body.\n\n{prompt}"
    # user_prompt = f"Please provide the implementation for the function body of `priority` in the following code. The goal is to maximize the size of the admissible set.\n\nFirst, inside <Thought> tags, briefly analyze the previous code and explain your new idea. Then, inside <Code> tags, provide ONLY the Python code for the function body.\n\n{prompt}"
    print(f"Prompt sent to LLM:\n---\n{prompt}\n---\n")
    
    retries = 5
    for attempt in range(retries):
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a search algorithm engineer. Your goal is to improve the computational logic of the function body. STRICTLY adhere to the function's signature and return type. DO NOT change the function's category or purpose. Write concise, high-performance code using branching structures or loops if necessary. Your response MUST be strictly formatted with <Thought> tags containing your reasoning, followed by <Code> tags containing ONLY the function body code. Do not repeat the function signature."},
                    {"role": "user", "content": user_prompt}
                ],
            )
            # if resp.usage:
            #     LLM._total_tokens_used += resp.usage.total_tokens
            # return resp.choices[0].message.content
            if resp.usage:
                LLM._total_tokens_used += resp.usage.total_tokens
            
            raw_content = resp.choices[0].message.content
            
            # 使用正则表达式提取 Thought 和 Code
            import re
            thought_match = re.search(r'<Thought>\s*(.*?)\s*</Thought>', raw_content, re.DOTALL)
            code_match = re.search(r'<Code>\s*(.*?)\s*</Code>', raw_content, re.DOTALL)
            
            if thought_match:
                print(f"\n[LLM Thought]:\n{thought_match.group(1)}\n")
            
            if code_match:
                # 如果成功提取到代码，只返回纯代码部分
                return code_match.group(1)
            else:
                # 容错机制：如果大模型没按格式输出，去掉可能存在的 markdown 代码块符号后直接返回
                cleaned_content = raw_content.replace('```python', '').replace('```', '')
                return cleaned_content
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
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)
    self._max_iterations = max_iterations
    self._start_time = time.time()

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
      
      elapsed_seconds = int(time.time() - self._start_time)
      
      print(f"Iteration: {iteration} | Total Tokens: {self._llm.total_tokens_used} | "
            f"Best Score (Island {prompt.island_id}): {best_score} | "
            f"Global Best: {global_best_score} | "
            f"Active Islands: {active_islands}/{num_islands} | "
            f"Elapsed: {elapsed_seconds}s | "
            f"Resets: {self._database._reset_count}")
# 👇👇👇 把下面这段加进去（用来把分数写入 csv 文件）
      import csv
      with open('score_log.csv', mode='a', newline='') as file:
          writer = csv.writer(file)
          # 记录：当前迭代次数，全局最高分
          writer.writerow([iteration, global_best_score])
      print("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
          if not island._clusters:
              continue
          print(f"  Island {i}: {len(island._clusters)} clusters")
          for sig, cluster in island._clusters.items():
              print(f"    Cluster {sig} (Score: {cluster.score}): {len(cluster._programs)} programs")
