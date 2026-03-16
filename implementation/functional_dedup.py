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

"""Functional duplicate detection for Sample-efficient FunSearch.

Tier 1: Syntax-level dedup via normalized code hash.
Tier 2: Function-level dedup via execution output (e.g. TSP tour).
"""
import hashlib


def _normalize_code_body(body: str) -> str:
  """Normalize code for Tier 1 hash: strip whitespace, comments, normalize."""
  if not body:
    return ""
  lines = body.splitlines()
  normalized = []
  for line in lines:
    stripped = line.strip()
    if stripped.startswith("#"):
      continue
    if stripped:
      normalized.append(stripped)
  return " ".join(normalized)


def code_body_hash(body: str) -> str:
  """Return hash of normalized code body for Tier 1 dedup."""
  normalized = _normalize_code_body(body)
  return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class FunctionalDedupCache:
  """Cache for functional duplicate detection.

  Tier 1: Code body hashes (syntax-level).
  Tier 2: (instance_id, tour_tuple) for TSP (function-level).
  """

  def __init__(self, tier1_only: bool = False) -> None:
    self._tier1_only = tier1_only
    self._evaluated_code_hashes: set[str] = set()
    self._evaluated_tour_signatures: set[tuple[str, tuple[int, ...]]] = set()

  def check_tier1(self, body: str) -> bool:
    """Return True if this code body was already evaluated (duplicate)."""
    h = code_body_hash(body)
    return h in self._evaluated_code_hashes

  def check_tier2(self, instance_id: str, tour: tuple[int, ...]) -> bool:
    """Return True if this (instance_id, tour) was already seen (duplicate)."""
    if self._tier1_only:
      return False
    sig = (instance_id, tour)
    return sig in self._evaluated_tour_signatures

  def add_tier1(self, body: str) -> None:
    """Record that this code body was evaluated."""
    self._evaluated_code_hashes.add(code_body_hash(body))

  def add_tier2(self, instance_id: str, tour: tuple[int, ...]) -> None:
    """Record that this (instance_id, tour) was evaluated."""
    if not self._tier1_only:
      self._evaluated_tour_signatures.add((instance_id, tour))

  def add_full_eval(self, body: str, instance_id: str, tour: tuple[int, ...]) -> None:
    """Record a full evaluation for future dedup."""
    self.add_tier1(body)
    self.add_tier2(instance_id, tour)
