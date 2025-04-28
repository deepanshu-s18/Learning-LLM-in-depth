"""
Minimal GRPO + RLVR building blocks for Chapter 7.

This file collects the small snippets shown in the chapter into one runnable
script. It is not a production trainer; it is a compact reference
implementation for the core mechanics:

1. deterministic verifier rewards
2. group-relative advantages
3. clipped GRPO loss with reference-policy KL
4. one training-step skeleton
"""

from __future__ import annotations

import re
from typing import Iterable

import torch


def extract_final_number(text: str) -> str | None:
