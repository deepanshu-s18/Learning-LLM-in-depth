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
