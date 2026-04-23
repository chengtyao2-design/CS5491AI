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
import json
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

  # def _draw_sample(self, prompt: str) -> str:
  #   """Returns a predicted continuation of `prompt`."""
  #   model = os.getenv("LLM_MODEL", "arcee-ai/trinity-large-preview:free")
  #   user_prompt = f"Please provide the implementation for the function body of `priority` in the following code. The goal is to maximize the size of the admissible set. \n\n{prompt}"
  #   print(f"Prompt sent to LLM:\n---\n{prompt}\n---\n")
    
  #   retries = 5
  #   for attempt in range(retries):
  #       try:
  #           resp = self.client.chat.completions.create(
  #               model=model,
  #               messages=[
  #                   {"role": "system", "content": "You are a search algorithm engineer. Your goal is to improve the computational logic of the function body. STRICTLY adhere to the function's signature and return type. DO NOT change the function's category or purpose. Write concise, high-performance code using branching structures or loops if necessary. Output code only, STRICTLY NO MARKDOWN and NO COMMENTS USING '#'. Your response should contain ONLY the function body code. Do not repeat the function signature or docstring."},
  #                   {"role": "user", "content": user_prompt}
  #               ],
  #           )
  #           if resp.usage:
  #               LLM._total_tokens_used += resp.usage.total_tokens
  #           return resp.choices[0].message.content
  #       except Exception as e:
  #           print(f"LLM API call failed (attempt {attempt+1}/{retries}): {e}")
  #           if attempt < retries - 1:
  #               time.sleep(2)
  #           else:
  #               print("Max retries reached. Stopping execution.")
  #               raise e
  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    # model = os.getenv("LLM_MODEL", "qwen/qwen-2.5-coder-32b-instruct")
    model = "openai/gpt-4o"

   
    system_prompt = (
        "You are an expert algorithm designer. Your goal is to discover new, mathematically "
        "innovative heuristic functions to maximize the size of the admissible set.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. FIRST, write a detailed 'Thoughts:' section.\n"
        "2. THEN, provide the Python implementation inside a ```python ... ``` code block.\n"
        "3. DO NOT repeat the `def priority(...)` signature. Write ONLY the indented function body.\n"
        "4. IMPORTANT FORMATTING: All lines of your code MUST have exactly the same baseline indentation (e.g., exactly 4 spaces). DO NOT mix indentation levels arbitrarily. IndentationError will cause immediate failure!\n"
        "5. INPUT TYPE: `el` is a `numpy.ndarray`. Use `np.count_nonzero(el == X)`, NEVER use list methods like `.count()`."
    )
    
    user_prompt = f"Please read the following context. Provide your Thoughts, then the new function body.\n\n{prompt}"
    
    import time
    retries = 5
    for attempt in range(retries):
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,   # <--- 强制解除 50 字封印
                temperature=0.9    # <--- 让模型发散思维
            )
            if resp.usage:
                LLM._total_tokens_used += resp.usage.total_tokens
                
            raw_content = resp.choices[0].message.content
            
            # import re
            # code_match = re.search(r'```python\s*(.*?)\s*```', raw_content, re.DOTALL)
            # if code_match:
            #     extracted_code = code_match.group(1)
            # else:
            #     code_match_fallback = re.search(r'```\s*(.*?)\s*```', raw_content, re.DOTALL)
            #     if code_match_fallback:
            #         extracted_code = code_match_fallback.group(1)
            #     else:
            #         # <--- 究极兜底：没代码就返回 0.0，绝不让英文报错
            #         extracted_code = "  return 0.0"
            import re
            
            # 终极正则：匹配 ```，忽略同行内容（不管有没有python或空格），匹配换行，然后提取代码！
            code_match = re.search(r'```[^\n]*\n(.*?)```', raw_content, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1)
            else:
                # 终极兜底：如果没有检测到代码块，返回保底得分
                extracted_code = "  return 0.0"
            
            lines = extracted_code.split('\n')
            final_lines = [line for line in lines if not line.strip().startswith('def priority')]
            final_code = '\n'.join(final_lines)
            
            print("\n" + "✨" * 20)
            print(f"💭 LLM Thoughts:\n{raw_content}\n")
            print(f"💻 Extracted Executable Code:\n{final_code}")
            print("✨" * 20 + "\n")
            
            return final_code
            
        except Exception as e:
            print(f"LLM API call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
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
        print("\n" + "="*20 + " LLM 生成的代码与思路展示 " + "="*20)
        print(sample)
        print("="*60 + "\n")
        
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
      # ---------- 新增：保存实验数据到 JSON ----------
      # 初始化一个列表用于记录历史数据（如果不存在的话）
      if not hasattr(self, '_experiment_history'):
          self._experiment_history = []
      
      # 处理 -inf，防止 JSON 序列化报错或格式不标准
      safe_best_score = best_score if best_score > -float('inf') else None
      safe_global_best = global_best_score if global_best_score > -float('inf') else None
      
      # 构建当前迭代的数据记录
      record = {
          "iteration": iteration,
          "island_id": prompt.island_id,
          "best_score": safe_best_score,
          "global_best_score": safe_global_best,
          "active_islands": active_islands,
          "total_islands": num_islands,
          "elapsed_seconds": elapsed_seconds,
          "total_tokens": self._llm.total_tokens_used
      }
      self._experiment_history.append(record)
      
      # 【关键修改】直接保存在当前运行目录下，不要再写 drive/MyDrive/ 了！
      save_path = "experiment_data.json" 
      with open(save_path, "w") as f:
          json.dump(self._experiment_history, f, indent=4)
      # -----------------------------------------------

      print("Cluster Stats:")
      for i, island in enumerate(self._database._islands):
          if not island._clusters:
              continue
          print(f"  Island {i}: {len(island._clusters)} clusters")
          for sig, cluster in island._clusters.items():
              print(f"    Cluster {sig} (Score: {cluster.score}): {len(cluster._programs)} programs")
