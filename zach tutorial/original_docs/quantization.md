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

## 1. The Mechanics of Mapping: From FP32 to INT8

We are now at the mathematical core of quantization. The goal is to represent a wide range of continuous floating-point numbers using a small, finite set of integers. This entire process is a simple mapping governed by two parameters: a **scale factor** and a **zero-point**.

#### The Starting Point: How a Float is Stored (FP32)

Computers represent decimal numbers using a standard format called **IEEE 754 single-precision floating-point**, or **FP32**. Every FP32 number consumes 32 bits, divided into three parts:

| Component | Bits | Purpose |
| :--- | :--- | :--- |
| **Sign (S)** | 1 bit | 0 for positive, 1 for negative. |
| **Exponent (E)**| 8 bits | Determines the number's magnitude (range). |
| **Mantissa (M)**| 23 bits | Stores the actual digits of the number (precision). |

The value is reconstructed using the formula: `Value = (-1)^S * (1 + Mantissa) * 2^(Exponent - 127)`

**Concrete Example: Representing the number `3.5`**

1.  **Sign (S):** The number is positive, so **`S = 0`**.
2.  **Binary Form:** `3.5` in binary is `11.1`.
3.  **Scientific Notation:** Normalize this to `1.11 x 2^1`.
4.  **Exponent (E):** The power is `1`. We add the bias `127`: `E = 1 + 127 = 128`. In 8-bit binary, this is **`10000000`**.
5.  **Mantissa (M):** The fractional part from the scientific notation is `.11`. We pad this to 23 bits: **`11000000000000000000000`**.

Putting it all together, `3.5` in FP32 is:
`[0] [10000000] [11000000000000000000000]`
(1 bit) (8 bits) (23 bits)

#### The Problem: FP32 is Too Big
