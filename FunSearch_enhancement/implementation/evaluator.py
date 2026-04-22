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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
import multiprocessing
import textwrap
from typing import Any

from implementation import code_manipulation
from implementation import programs_database


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''

  print(f"DEBUG: Raw generated code:\n{generated_code!r}")

  # Filter out full-line comments starting with #
  lines = generated_code.splitlines()
  lines = [line for line in lines if not line.strip().startswith('#')]
  generated_code = '\n'.join(lines)

  # Remove markdown code blocks if present
  if '```' in generated_code:
    lines = generated_code.splitlines()
    # Filter out lines that start with ```
    lines = [line for line in lines if not line.strip().startswith('```')]
    generated_code = '\n'.join(lines)
    print(f"DEBUG: Cleaned markdown code:\n{generated_code!r}")

  # Try to detect if the code is a full function definition (Issue 1)
  try:
    # Attempt to parse directly without wrapper first
    direct_tree = ast.parse(generated_code)
    # Check if it contains a single FunctionDef at top level
    func_defs = [node for node in ast.walk(direct_tree) if isinstance(node, ast.FunctionDef)]
    if func_defs and len(func_defs) >= 1:
       # Use the first function found (assuming it's the intended one)
       print("DEBUG: Detected full function definition, extracting body...")
       target_node = func_defs[0]

       if target_node.body:
           body_start_line = target_node.body[0].lineno

           lines = generated_code.splitlines()
           pass 
  except SyntaxError:
    pass

  # Ensure the code is indented for parsing
  lines = generated_code.splitlines()
  # Find first non-empty line to check indentation
  first_line_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
  
  if first_line_idx is not None:
      first_line = lines[first_line_idx]
      # Check if it looks like a function definition
      if first_line.strip().startswith('def ') and ':' in first_line:
           print("DEBUG: Detected function signature in first line. Attempting to strip it.")
           
           # Calculate indentation of the def
           def_indent = len(first_line) - len(first_line.lstrip())
           
           # Find where the function body ends (by indentation)
           # This is complex. Let's just try to parse it as is (it's a valid function).
           try:
               tree = ast.parse(generated_code)
               # If successful, extract body
               for node in ast.walk(tree):
                   if isinstance(node, ast.FunctionDef):
                       # Found it.
                       body_nodes = node.body
                       if body_nodes:
                           start_line = body_nodes[0].lineno - 1
                           end_line = node.end_lineno
                           body_segment = lines[start_line:end_line]
                           # Dedent this segment
                           segment_str = '\n'.join(body_segment)
                           dedented = textwrap.dedent(segment_str)
                           print("DEBUG: Successfully extracted body from nested function.")
                           generated_code = dedented
                           lines = generated_code.splitlines()
                           first_line_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
                           if first_line_idx is not None:
                               first_line = lines[first_line_idx]
                           break
           except SyntaxError:
               print("DEBUG: Failed to parse potential function definition, treating as body.")

      # Re-calculate indentation after potential stripping
      initial_indent = len(first_line) - len(first_line.lstrip())
      
      # Normalize indentation: remove 'initial_indent' from all lines to align left
      if initial_indent > 0:
          normalized_lines = []
          for i, line in enumerate(lines):
              if not line.strip():
                  normalized_lines.append('')
                  continue
              
              current_indent = len(line) - len(line.lstrip())
              if current_indent >= initial_indent:
                  normalized_lines.append(line[initial_indent:])
              else:
                  # Line is less indented than start (e.g. outlier), strip it to 0
                  normalized_lines.append(line.lstrip())
          generated_code = '\n'.join(normalized_lines)

  code = f'def fake_function_header():\n' + textwrap.indent(generated_code, '  ')
  
  print(f"DEBUG: Wrapped code for parsing:\n{code}")

  tree = None
  
  # Helper to try parsing
  def _attempt_parse(code_str):
      try:
          return ast.parse(code_str)
      except SyntaxError:
          return None

  # Try 1: Parse original wrapped code
  tree = _attempt_parse(code)
  
  # Try 2: If failed, try to fix inconsistent indentation (replace 4 spaces with 2 spaces)
  if tree is None:
      print("DEBUG: Parse failed, attempting indentation fix...")
      
      # Strategy: Quantize indentation to multiples of 2
      fixed_lines = []
      for line in generated_code.splitlines():
          stripped = line.lstrip()
          if not stripped:
              fixed_lines.append(line)
              continue
              
          indent = len(line) - len(stripped)
          
          if indent > 0 and indent % 4 == 0:
              new_indent = indent // 2
              fixed_lines.append(' ' * new_indent + stripped)
          else:
              fixed_lines.append(line)
      
      fixed_generated_code = '\n'.join(fixed_lines)
      
      # Re-wrap
      fixed_code = f'def fake_function_header():\n' + textwrap.indent(fixed_generated_code, '  ')
          
      tree = _attempt_parse(fixed_code)
      if tree is not None:
          code = fixed_code
          print("DEBUG: Indentation fix successful (4->2)")

  # Try 3: Repair loop for specific lines (Issue 2)
  if tree is None:
      print("DEBUG: Parse failed, entering repair loop...")
      current_code = code
      max_repairs = 5
      for attempt in range(max_repairs):
          try:
              tree = ast.parse(current_code)
              code = current_code
              print(f"DEBUG: Repair successful on attempt {attempt}")
              break
          except SyntaxError as e:
              # Try to fix the specific line
              lineno = e.lineno
              if lineno is None: 
                  break
              
              lines = current_code.splitlines()
              if lineno > len(lines):
                  break
                  
              bad_line_idx = lineno - 1
              bad_line = lines[bad_line_idx]
              
              print(f"DEBUG: SyntaxError at line {lineno}: {bad_line.strip()}")
              
              # Heuristic: Dedent the line if it looks like control flow
              # (e.g. 'if', 'else', 'elif', 'return', 'for') that might be mis-indented
              stripped = bad_line.lstrip()
              indent = len(bad_line) - len(stripped)
              
              if indent >= 2:
                  # Try dedenting by 2 spaces
                  new_line = ' ' * (indent - 2) + stripped
                  lines[bad_line_idx] = new_line
                  current_code = '\n'.join(lines)
                  print(f"DEBUG: Attempting to dedent line {lineno}")
              else:
                  # Can't dedent further, break to truncation
                  break

  # We keep trying and deleting code from the end until the parser succeeds.

  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      print(f"DEBUG: SyntaxError at line {e.lineno}, truncating...")
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    print("DEBUG: Code became empty after truncation")
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  print(f"DEBUG: Function end line detected: {visitor.function_end_line}")
  
  body_lines = code.splitlines()[1:visitor.function_end_line]
  print(f"DEBUG: Extracted body lines:\n{body_lines}")
  
  # Normalize indentation to 2 spaces
  body = '\n'.join(body_lines)
  body = textwrap.dedent(body)
  body = textwrap.indent(body, '  ')
  
  return body + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  body = _trim_function_body(generated_code)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)


def _run_in_subprocess(program: str, function_name: str, test_input: Any, queue: multiprocessing.Queue):
  """Executes the generated code in a subprocess."""
  try:
    # Use the same dictionary for globals and locals to ensure that functions
    # defined in `program` can access imports defined in `program`.
    local_scope = {}
    # exec() executes the program in the local_scope.
    exec(program, local_scope, local_scope)
    if function_name not in local_scope:
      queue.put((None, False))
      return
    func = local_scope[function_name]
    try:
      result = func(test_input)
      if isinstance(result, (list, tuple)) or hasattr(result, '__len__') or hasattr(result, 'shape'):
          # If result is array-like, check if it's empty or handle ambiguous truth value
          # Assuming we expect a scalar score, try to convert or summarize
          import numpy as np
          if isinstance(result, np.ndarray):
             if result.size == 1:
                 result = result.item()
             else:
                 try:
                    result = float(result)
                 except:
                    # Fallback for array output when scalar expected
                    pass
    except ValueError as e:
      if "The truth value of an array" in str(e):
         print(f"Caught ValueError in user code: {e}")
         result = None
      else:
         raise e
         
    queue.put((result, True))
  except Exception:  # pylint: disable=broad-except
    # Print exception for debugging purposes
    import traceback
    traceback.print_exc()
    queue.put((None, False))


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: Any,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_in_subprocess,
        args=(program, function_to_run, test_input, queue),
    )
    process.start()
    try:
      result, success = queue.get(timeout=timeout_seconds)
      process.join()
      return result, success
    except Exception:  # pylint: disable=broad-except
      # If the process times out or fails for other reasons.
      if process.is_alive():
        process.terminate()
        process.join()
      return None, False


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)

    scores_per_test = {}
    for current_input in self._inputs:
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)
