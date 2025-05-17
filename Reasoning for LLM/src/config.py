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
MODEL_SIZES = {
    "Flan-T5 Small": "80M",
    "Flan-T5 Base": "250M",
    "Flan-T5 Large": "800M",
    "Zephyr-7B": "7B",
    "Phi-2": "2.7B",
    "TinyLlama-1.1B": "1.1B",
}

# Generation parameters
GEN_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.3,
    "top_p": 0.95,
}

# Default sample limits
DEFAULT_SAMPLE_LIMIT = 50
PREDICTION_PREVIEW_LIMIT = 5

