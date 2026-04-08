# Quantization in 20 Min
> Quantization is Easy

## Introduction: Decoding the Black Box of Quantization

You've seen the terms: **4-bit, INT8, per-channel, group size, PTQ**. You know they shrink large models, but the underlying mechanism—the *how* and the *why*—remains a black box. You want to move past using libraries as magic and grasp the engineering principles that make them work.

This tutorial decodes that black box. This is not a guide on which library function to call. It is a direct, concise walkthrough of the core algorithms and design choices behind modern weights-only quantization.

**In the next 40 minutes, you will build a first-principles understanding of:**
*   **The Quantization Algorithm:** The fundamental affine transformation that maps floats to integers.
*   **Core Parameters:** What **Scale (S)** and **Zero-Point (Z)** actually represent and how they are calculated.
*   **Granularity & Precision:** The critical trade-off between **Per-Tensor** and **Per-Channel** quantization and why it dictates model accuracy.
*   **The 4-bit Standard:** Why **Group Size** (e.g., block size 128) is essential for preventing catastrophic quality loss in low-bit formats.
*   **Deployment Methods:** The practical difference between **Post-Training Quantization (PTQ)**, the default industry method, and its alternative, **Quantization-Aware Training (QAT)**.

You will learn to command the fundamental formulas that make this compression possible.

#### The Core Algorithm You Will Master

1.  **Quantization (Float -> Integer):**
    $$q = \text{clamp}\left(\text{round}\left(\frac{x}{S} + Z\right), q_{min}, q_{max}\right)$$

2.  **Dequantization (Integer -> Float):**
    $$\hat{x} = S(q - Z)$$

By the end, these equations will not be abstract symbols. You will understand their components, their impact on hardware, and the rationale behind their application. Let's begin.

