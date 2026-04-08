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

This representation is precise but inefficient for inference.
*   **Memory Cost:** Each parameter takes 32 bits, or 4 bytes. For a 7-billion parameter model:
    `7,000,000,000 parameters * 4 bytes/parameter = 28,000,000,000 bytes = 28 GB of VRAM.`
*   **Bandwidth Cost:** The GPU is starved waiting for these 28 GB of weights to be moved from VRAM to the compute cores. This memory transfer is often the true bottleneck.

#### The Solution: Mapping to a Simpler Format (INT8)

Our goal is to convert these complex 32-bit structures into simple **8-bit signed integers (INT8)**. An INT8 is just a standard 8-bit number, representing a direct integer value from `-128` to `127`. For example, the integer `44` is simply `00101100`. There is no sign, exponent, or mantissa.

This is a **4x reduction** in size. The challenge is to create a mathematical map from the FP32 world to the INT8 world without losing critical information.

#### Building Intuition: Scale and Zero-Point

Imagine you have a small tensor of weights that fall within the range `[-3.5, 3.5]`.

`weights_fp32 = [1.2, -3.5, 0.8, 2.1, -1.9, 3.5]`

Our task is to represent these numbers using INT8, which can only hold values between `[-128, 127]`. How do we map every possible number from `-3.5` to `3.5` onto one of the 256 available integer "buckets"?

1.  **Calculate the Scale (S):** The scale is our "step size." It tells us how many float units correspond to a single integer unit.
    *   The total span of our float values is `3.5 - (-3.5) = 7.0`.
    *   The total span of our INT8 values is `127 - (-128) = 255`.
    *   **Scale (S) = Float Range / Integer Range = 7.0 / 255 ≈ 0.02745**

    This means for every step we take in the integer world (e.g., from 10 to 11), we are moving approximately `0.02745` in the float world.

2.  **Calculate the Zero-Point (Z):** The zero-point is an offset or "shift." It ensures that the floating-point value `0.0` maps correctly to an integer. Since our float range `[-3.5, 3.5]` is perfectly symmetric around zero, we can align its zero with the integer zero.
    *   **Zero-Point (Z) = 0**
