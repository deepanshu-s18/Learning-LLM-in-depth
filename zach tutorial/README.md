# Zach's Interactive LLM Mastery Series 🚀

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Pocket/PocketFlow-Tutorial-Video-Generator/blob/main/docs/llm/00_Tutorial_Roadmap_and_Index.ipynb)

Welcome to **Zach's LLM Mastery Tutorial Series** from PocketFlow, fully converted into **15 interactive, runnable Google Colab / Jupyter notebooks**! 

Every notebook contains:
1. Complete narrative explanations, intuitive analogies, and LaTeX math formulas.
2. Step-by-step implementations built from scratch with pure Python, NumPy, and PyTorch.
3. Interactive visualizations, loss curves, decision boundaries, attention heatmaps, and speed benchmarks.
4. Interactive Playground cells at the end of each topic for hands-on experimentation with hyperparameters.

---

## 📚 Complete Curriculum & Notebook Directory

| # | Interactive Notebook | Original Doc | Focus Area | Key Concepts & Implementations |
|---|---|---|---|---|
| **00** | [00_Tutorial_Roadmap_and_Index.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/00_Tutorial_Roadmap_and_Index.ipynb) | - | Master Index | Full curriculum roadmap, prerequisites & environment check |
| **01** | [01_Neural_Networks_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/01_Neural_Networks_From_Scratch.ipynb) | [`nn.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/nn.md) | DL Foundations | Gradient Descent, Chain Rule, Backprop from scratch in NumPy, Non-linear Decision Boundary Plot |
| **02** | [02_PyTorch_Deep_Dive.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/02_PyTorch_Deep_Dive.ipynb) | [`pytorch.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/pytorch.md) | PyTorch Core | Tensors, Autograd computation graphs, `nn.Module`, Loss functions, Optimizers, Production Training Loop |
| **03** | [03_Adam_Optimizer_Demystified.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/03_Adam_Optimizer_Demystified.ipynb) | [`adam.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/adam.md) | Optimization | SGD vs Momentum vs RMSprop vs Adam, Bias correction, 2D Ravine Contour Optimization Trajectories |
| **04** | [04_Attention_Mechanism_Step_by_Step.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/04_Attention_Mechanism_Step_by_Step.ipynb) | [`attention.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/attention.md) | Attention | Scaled Dot-Product Attention ($Q, K, V$), Softmax $\sqrt{d_k}$ scaling, Multi-Head Attention, Alignment Heatmap |
| **05** | [05_Transformer_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/05_Transformer_From_Scratch.ipynb) | [`transformer.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/transformer.md) | GPT Architecture | Complete Decoder-Only Transformer, Pre-LN Blocks, Causal Mask, Character text generation |
| **06** | [06_KV_Cache_Optimization.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/06_KV_Cache_Optimization.ipynb) | [`kv_cache.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/kv_cache.md) | Inference Speed | Autoregressive decoding bottleneck, KV cache buffers, Generation Speed Benchmark (Naive vs KV Cache) |
| **07** | [07_Rotary_Positional_Encoding_RoPE.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/07_Rotary_Positional_Encoding_RoPE.ipynb) | [`rope.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/rope.md) | Modern LLMs | 2D Complex Rotation Matrix, High-dimensional RoPE, Relative Distance Invariance verification |
| **08** | [08_LLM_Pretraining_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/08_LLM_Pretraining_From_Scratch.ipynb) | [`pretrain.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/pretrain.md) | Pre-training | Next-token prediction, Self-supervised learning, Causal Cross-Entropy Loss, Perplexity (PPL) tracking |
| **09** | [09_Supervised_Fine_Tuning_SFT.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/09_Supervised_Fine_Tuning_SFT.ipynb) | [`sft.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/sft.md) | Instruction Tuning | Turning base models into assistants, Chat formatting, Prompt-Response Loss Masking (`label = -100`) |
| **10** | [10_LoRA_Low_Rank_Adaptation.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/10_LoRA_Low_Rank_Adaptation.ipynb) | [`lora.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/lora.md) | Efficient Fine-Tuning | PEFT, Low-Rank Decomposition $W + \frac{\alpha}{r}BA$, Freezing base model, Zero-overhead weight merging |
| **11** | [11_LLM_Quantization_INT8_INT4.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/11_LLM_Quantization_INT8_INT4.ipynb) | [`quantization.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/quantization.md) | Compression | INT8 & INT4 formats, Scale and Zero-Point calibration, Quantization error SNR/MSE measurement |
| **12** | [12_RLHF_and_PPO_Alignment.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/12_RLHF_and_PPO_Alignment.ipynb) | [`rlhf.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/rlhf.md) | Preference Alignment | Bradley-Terry Reward Modeling, PPO Clipped Surrogate Loss, KL Divergence penalty against reference policy |
| **13** | [13_Direct_Preference_Optimization_DPO.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/13_Direct_Preference_Optimization_DPO.ipynb) | [`dpo.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/dpo.md) | Modern Alignment | Direct alignment from preference pairs, Eliminating the Reward Model & RL loop, Implicit reward tracking |
| **14** | [14_Diffusion_Models_From_Scratch.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/14_Diffusion_Models_From_Scratch.ipynb) | [`diffusion.md`](file:///Users/shobhitagnihotri/Desktop/internship/zach%20tutorial/diffusion.md) | Generative Models | DDPM Forward Gaussian Noise Schedule ($\beta_t, \alpha_t, \bar{\alpha}_t$), UNet Noise Predictor, Reverse Denoising Sampling |

---

## 💻 How to Run

### Option 1: Run in Google Colab (Cloud GPU)
Click on any of the **Open In Colab** badges at the top of each notebook or open them directly in [Google Colab](https://colab.research.google.com/).

### Option 2: Run Locally (VS Code / JupyterLab)
1. Install dependencies in your local Python environment:
   ```bash
   pip install torch torchvision matplotlib numpy scipy jupyterlab
   ```
2. Launch JupyterLab or open this workspace in VS Code:
   ```bash
   cd "zach tutorial"
   jupyter lab
   ```
3. Open any `.ipynb` notebook and execute cells with `Shift + Enter`!

---

## 📁 Folder Structure
```
zach tutorial/
├── 00_Tutorial_Roadmap_and_Index.ipynb
├── 01_Neural_Networks_From_Scratch.ipynb
├── 02_PyTorch_Deep_Dive.ipynb
├── 03_Adam_Optimizer_Demystified.ipynb
├── 04_Attention_Mechanism_Step_by_Step.ipynb
├── 05_Transformer_From_Scratch.ipynb
├── 06_KV_Cache_Optimization.ipynb
├── 07_Rotary_Positional_Encoding_RoPE.ipynb
├── 08_LLM_Pretraining_From_Scratch.ipynb
├── 09_Supervised_Fine_Tuning_SFT.ipynb
├── 10_LoRA_Low_Rank_Adaptation.ipynb
├── 11_LLM_Quantization_INT8_INT4.ipynb
├── 12_RLHF_and_PPO_Alignment.ipynb
├── 13_Direct_Preference_Optimization_DPO.ipynb
├── 14_Diffusion_Models_From_Scratch.ipynb
├── notebooks/                  # Mirrored copies of all notebooks
├── original_docs/              # Preserved original markdown and html files
├── README.md                   # This master documentation guide
└── generate_complete_suite.py  # Automation script for notebook generation
```
