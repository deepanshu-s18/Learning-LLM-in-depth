"""
Configuration settings and model registries for LLM reasoning evaluations.
"""

# Seq2Seq models evaluated with CoT
SEQ2SEQ_MODELS = {
    "Flan-T5 Small": "google/flan-t5-small",
    "Flan-T5 Base": "google/flan-t5-base",
    "Flan-T5 Large": "google/flan-t5-large",
}

# Causal / Decoder models evaluated with CoT
DECODER_MODELS = {
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-alpha",
    "Phi-2": "microsoft/phi-2",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Model parameter size mappings for visualization
