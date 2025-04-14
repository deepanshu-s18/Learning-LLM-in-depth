
# Chapter 5: Multi-Token Prediction and FP8 Quantization

This chapter covers two key innovations for improving DeepSeek's training and inference efficiency. First, we explore Multi-Token Prediction (MTP), a technique that provides stronger training signals by predicting multiple future tokens simultaneously. Second, we dive into DeepSeek's advanced FP8 quantization framework, which trades precision for significant gains in speed and memory without sacrificing performance.

### Main Chapter Code

- [Chapter_5.ipynb](01_main-chapter-code/Chapter_5.ipynb) — Contains the from-scratch implementations of:
  - **RMSNorm**: Root Mean Square Layer Normalization (Listing 5.1)
  - **DeepSeekMTPModule**: The causal MTP module with merge-project-transform pipeline (Listing 5.2)
  - **DeepSeekV3WithMTP**: Full model with shared Transformer trunk and MTP chain (Listings 5.3 & 5.4)
  - **Verification**: End-to-end test of the MTP architecture (Listing 5.5)
