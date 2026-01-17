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
                )

            output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            pred = extract_predicted_answer(output_text)

            if is_answer_match(pred, gt_answer):
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"[{self.model_id}] Accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    def preview_predictions(
        self,
        dataset: Dataset,
        prefix: str = FEW_SHOT_COT_PREFIX,
        num_samples: int = 5
    ) -> None:
        """Print sample step-by-step reasoning outputs."""
        print(f"\n--- Predictions from {self.model_id} on {num_samples} Questions ---")
        subset = dataset.select(range(min(num_samples, len(dataset))))

        for i, sample in enumerate(subset):
            question = sample["question"]
            gt_answer = extract_ground_truth_answer(sample["answer"])

            prompt = format_cot_prompt(question, prefix=prefix)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=128
                )

            output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            pred = extract_predicted_answer(output_text)

            print(f"\nExample {i + 1}")
            print(f"Q: {question}")
            print(f"Ground Truth: {gt_answer}")
            print(f"Generated CoT Output:\n{output_text}")
            print(f"Extracted Answer: {pred} | Match: {is_answer_match(pred, gt_answer)}")


class DecoderEvaluator:
    """Evaluator for Causal / Autoregressive models (Zephyr, Phi-2, TinyLlama)."""

    def __init__(self, model_id: str, torch_dtype: Optional[torch.dtype] = None):
        self.model_id = model_id
        print(f"Loading Decoder model '{model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=False
        )

    def evaluate(
        self,
        dataset: Dataset,
        prefix: str = FEW_SHOT_COT_PREFIX
    ) -> float:
        """Run few-shot CoT evaluation on dataset."""
        print(f"\nEvaluating Decoder {self.model_id}...")
        correct = 0
        total = 0

        for sample in dataset:
            question = sample["question"]
            gt_answer = extract_ground_truth_answer(sample["answer"])

            prompt = format_cot_prompt(question, prefix=prefix)
            outputs = self.pipe(prompt)
            generated_full = outputs[0]["generated_text"]
            output_text = generated_full.split("A:")[-1].strip()
            pred = extract_predicted_answer(output_text)

            if is_answer_match(pred, gt_answer):
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"[{self.model_id}] Accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    def preview_predictions(
        self,
        dataset: Dataset,
        prefix: str = FEW_SHOT_COT_PREFIX,
        num_samples: int = 5
    ) -> None:
        """Print detailed reasoning trajectories."""
        print(f"\n--- Predictions from {self.model_id} on {num_samples} Questions ---")
        subset = dataset.select(range(min(num_samples, len(dataset))))

        for i, sample in enumerate(subset):
            question = sample["question"]
            gt_answer = extract_ground_truth_answer(sample["answer"])

            prompt = format_cot_prompt(question, prefix=prefix)
            outputs = self.pipe(prompt)
            generated_full = outputs[0]["generated_text"]
            output_text = generated_full.split("A:")[-1].strip()
            pred = extract_predicted_answer(output_text)

            print(f"\nExample {i + 1}")
            print(f"Q: {question}")
            print(f"Ground Truth: {gt_answer}")
            print(f"Generated CoT Output:\n{output_text}")
            print(f"Extracted Answer: {pred} | Match: {is_answer_match(pred, gt_answer)}")
