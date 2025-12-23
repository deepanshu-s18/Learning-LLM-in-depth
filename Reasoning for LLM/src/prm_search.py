"""
Process Reward Model (PRM) Guided Beam Search for Step-by-Step LLM Reasoning.

Implements inference-time compute scaling via search over reasoning steps,
scoring partial reasoning traces using a PRM / sequence classification reward model,
and visualizing the resulting reasoning tree graph.
"""

import os
import textwrap
import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def get_default_device():
