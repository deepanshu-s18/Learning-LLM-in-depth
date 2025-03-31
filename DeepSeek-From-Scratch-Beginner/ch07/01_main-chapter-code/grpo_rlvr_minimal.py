"""
GRPO + RLVR minimal implementation for chapter 7.
"""

import re
import torch


def get_last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None
