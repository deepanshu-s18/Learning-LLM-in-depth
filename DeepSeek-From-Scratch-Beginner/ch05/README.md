
# Chapter 5: Multi-Token Prediction and FP8 Quantization

This chapter covers two key innovations for improving DeepSeek's training and inference efficiency. First, we explore Multi-Token Prediction (MTP), a technique that provides stronger training signals by predicting multiple future tokens simultaneously. Second, we dive into DeepSeek's advanced FP8 quantization framework, which trades precision for significant gains in speed and memory without sacrificing performance.

### Main Chapter Code

- [Chapter_5.ipynb](01_main-chapter-code/Chapter_5.ipynb) — Contains the from-scratch implementations of:
  - **RMSNorm**: Root Mean Square Layer Normalization (Listing 5.1)
  - **DeepSeekMTPModule**: The causal MTP module with merge-project-transform pipeline (Listing 5.2)
  - **DeepSeekV3WithMTP**: Full model with shared Transformer trunk and MTP chain (Listings 5.3 & 5.4)
  - **Verification**: End-to-end test of the MTP architecture (Listing 5.5)
  - **FP8 Quantization**: Basic scaling, fine-grained tile-wise quantization, online vs delayed scaling, and FP8 linear layer

### Relevant Videos for this Chapter

- **Multi-Token Prediction (MTP):**
  - [Multi-Token Prediction Introduction](https://www.youtube.com/watch?v=tMtHAAg0UT4)
  - [How DeepSeek rewrote Multi-Token Prediction (MTP)?](https://www.youtube.com/watch?v=4GmwJLvwaXE)
  - [Multi Token Prediction (MTP) Coded from Scratch](https://www.youtube.com/watch?v=lyHe8_JHoVI)
- **Quantization:**
  - [Introduction to LLM Quantization](https://www.youtube.com/watch?v=0U9l3-r6jVE)
  - [How DeepSeek Rewrote Quantization Part 1 | Mixed Precision | Fine-grained quantization](https://www.youtube.com/watch?v=xftka2aXnm4)
  - [How DeepSeek Rewrote Quantization Part 2 | Accumulation Precision | Online Quantization](https://www.youtube.com/watch?v=FxDbrWBENy8)
