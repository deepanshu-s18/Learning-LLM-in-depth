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
            self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        dataset: Dataset,
        prefix: str = FEW_SHOT_COT_PREFIX,
        max_new_tokens: int = 128
    ) -> float:
        """Run few-shot CoT evaluation over dataset."""
        print(f"\nEvaluating Seq2Seq {self.model_id}...")
        correct = 0
        total = 0

        for sample in dataset:
            question = sample["question"]
            gt_answer = extract_ground_truth_answer(sample["answer"])

            prompt = format_cot_prompt(question, prefix=prefix)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
