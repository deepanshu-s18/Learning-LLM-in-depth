"""
Evaluation engines for Seq2Seq and Causal/Decoder Language Models.
"""

import re
import torch
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from datasets import Dataset

from src.prompts import format_cot_prompt, FEW_SHOT_COT_PREFIX
from src.dataset import extract_ground_truth_answer, extract_predicted_answer, is_answer_match


class Seq2SeqEvaluator:
    """Evaluator for encoder-decoder models like FLAN-T5."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Seq2Seq model '{model_id}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        if self.device != "cpu":
