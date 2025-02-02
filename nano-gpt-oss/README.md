<div align="center">

# Nano-GPT-OSS Language Model

**An open-source transformer that balances full-context and sliding-window attention for efficient, scalable LLM training and inference.**

<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://huggingface.co"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFC107?logo=hugging%20face&logoColor=black" alt="Hugging Face"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>

![Val Loss of Gpt oss](assets/val-loss.png)

</div>

## Training & Validation Loss

| loss | Training Loss| Validation Loss | Num Heads | Trf BLock | Hidden Dim |
|--------|---------|---------|----------|----------|----------|
| GPT-OSS | **1.981** | **1.682** | 12 | 12 | 1020 |
| GPT2 | 3.124 | 2.747 | 12 | 12 | 1020 |
| GPT-OSS | **2.034** | **1.725** | 12 | 8 | 1020 |
| GPT2 | 2.593 | 2.173 | 12 | 8 | 1020 |
| GPT-OSS | **2.031** | **1.778** | 12 | 6 | 1020 |
| GPT2 | 2.570 | 2.331 | 12 | 6 | 1020 |
| GPT-OSS | **1.984** | **1.678** | 8 | 12 | 1024 |
| GPT2 | 2.445 | 2.036 | 8 | 12 | 1024 |
| GPT-OSS | **2.212** | **1.901**| 8 | 8 | 1024 |
| GPT2 | 2.416 | 2.011 | 8 | 8 | 1024 |
