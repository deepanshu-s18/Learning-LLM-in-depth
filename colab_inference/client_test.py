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

    response_stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert AI and LLM inference engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
        stream=True
    )

    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.time() - start_time
            print(token, end="", flush=True)
            token_count += 1

    total_time = time.time() - start_time
    tokens_per_sec = token_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("📊 INFERENCE BENCHMARK METRICS:")
    print(f"  • Time to First Token (TTFT): {first_token_time:.2f}s" if first_token_time else "  • TTFT: N/A")
    print(f"  • Total Time:                 {total_time:.2f}s")
    print(f"  • Generated Tokens:           {token_count}")
    print(f"  • Generation Speed:           {tokens_per_sec:.2f} tokens/sec")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()
