"""
client_test.py
==============
Test your Colab vLLM inference server locally from Antigravity/Mac.
Measures tokens/second, time to first token (TTFT), and stream output.
"""

import time
from openai import OpenAI

# 🔴 Replace with the public ngrok URL printed by Colab Notebook 2
# Example: "https://a1b2-34-123.ngrok-free.app/v1"
NGROK_BASE_URL = "https://YOUR-NGROK-SUBDOMAIN.ngrok-free.app/v1"

# The model path as loaded in vLLM
MODEL_NAME = "/content/drive/MyDrive/LLM_Model_Hub/open_source_models/casperhansen--llama-3-8b-instruct-awq"

def test_inference(prompt: str = "Explain Grouped-Query Attention (GQA) and why it saves KV-cache memory."):
    print(f"Connecting to: {NGROK_BASE_URL} ...\n")
    client = OpenAI(
        base_url=NGROK_BASE_URL,
        api_key="sk-no-key-required-for-self-hosted"
    )

    print(f"👤 User: {prompt}\n")
    print("🤖 Response: ", end="", flush=True)

    start_time = time.time()
    first_token_time = None
    token_count = 0

