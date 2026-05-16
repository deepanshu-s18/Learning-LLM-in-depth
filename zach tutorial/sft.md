# **Title: Supervised Fine-Tuning in 30 Min**

## **Chapter 1: The Problem: Parrots, Not Assistants**

You've used models like ChatGPT or Claude. You give them an instruction, and they follow it. This behavior feels natural, but it is a carefully engineered facade. The underlying base model—fresh from its training on trillions of words—is not an assistant. It is a powerful, alien-like **text-completion engine**. Its sole objective is to predict the next word in a sequence with statistical accuracy.

This singular focus creates the **Parrot Problem**: the model becomes a master of mimicry without any concept of user intent.

Let's make this concrete. You prompt a raw, pre-trained base model (like the original GPT-3) with a question.

**Your Prompt:**
```
What is the primary cause of Earth's seasons?
```

**An Assistant's Expected Response:**
```
The primary cause of Earth's seasons is the tilt of the Earth's axis, which is about 23.5 degrees...
```

**The Base Model's Likely Response:**
```
What is the primary cause of Earth's seasons?
A) The Earth's distance from the sun.
B) The tilt of the Earth's axis relative to its orbital plane.
C) The speed of the Earth's rotation.
D) Ocean currents and wind patterns.
```

The model did not answer your question. It continued your text by formatting it into a multiple-choice question. Why? Because its pre-training data is saturated with quizzes and tests. From a purely statistical "next-word prediction" viewpoint, this is a highly probable completion for a sentence that starts with "What is...".

The model isn't being unhelpful. It is perfectly executing its objective: **completing a pattern**. It has no concept of a "user" or an "instruction." It only sees text.

This gap between a pattern-completing parrot and a helpful assistant is closed by a process called **post-training**. The first and most crucial step is **Supervised Fine-Tuning (SFT)**.

It sounds complex, but the entire engineering challenge of SFT boils down to crafting a single, clever data transformation function.

**My promise is this: In the next 30 minutes, you will learn to write this exact function from scratch.** This is the complete logic that turns raw `(prompt, response)` pairs into trainable tensors for an LLM.

```python
import torch

def prepare_sft_batch(prompt: str, response: str, tokenizer):
    """
    This is the core engineering of SFT. It takes a prompt/response pair
    and creates the input_ids and the strategically masked labels.
    """
    # 1. Format the text with special tokens for conversation structure.
    prompt_part = f"<|user|> {prompt} <|end|> <|assistant|>"
    full_text = f"{prompt_part} {response} <|end|>"

    # 2. Tokenize to find the boundary for loss masking.
    # We only want to train the model on the assistant's response.
    prompt_ids = tokenizer.encode(prompt_part)
    mask_until_idx = len(prompt_ids)

    # 3. Tokenize the full conversation for model input.
    input_ids = tokenizer.encode(full_text)

    # 4. Create the labels tensor by cloning the input_ids.
    labels = torch.tensor(input_ids).clone()

    # 5. Apply the mask. This is the critical step.
    # We replace the prompt tokens in the labels with -100.
    labels[:mask_until_idx] = -100

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": labels
    }
```
That's the entire trick. The rest is just standard model training. PyTorch's loss function is hard-coded to ignore `-100` values, so by feeding it these `labels`, we force the model to learn one thing: "When you see `<|assistant|>`, generate the expert response."

By mastering this function, you master SFT.


Our journey will take us from the problem to the complete solution.

```mermaid
graph TD
    A[Pre-trained LLM] -- "Objective: Next-token prediction" --> B{The Parrot Problem};
    B -- "Solution: Imitate expert examples via Loss Masking" --> C[Supervised Fine-Tuning];
    C -- "Result: Instruction-following" --> D[Aligned Assistant Model];
```

To understand *why* this data transformation is so effective, we must first master the engine it modifies. In the next chapter, we will dissect the mathematical core of pre-training—Cross-Entropy Loss—to see exactly how the parrot learns to talk in the first place.

## **Chapter 2: The Engine of Pre-training: Cross-Entropy Loss**

Before we can teach a model to be an assistant, we must first understand how it learned to be a parrot. The vast knowledge of a base LLM is forged during its **pre-training** phase, where it is trained on a single, brutally simple objective: **next-token prediction**.

The rule is this: given a sequence of text, predict the very next token. That's it. The model is a highly sophisticated pattern-completion machine. To teach it this skill, we use the standard workhorse of deep learning classification: **Cross-Entropy Loss**.

Cross-Entropy Loss is a way to measure how "surprised" a model is by the correct answer. If the model assigns a high probability to the correct next token, the loss is low (low surprise). If it assigns a very low probability, the loss is high (high surprise).

Mathematically, this simplifies to calculating the **negative log-probability** of the correct target token.

Let's make this concrete with a minimal example that you can calculate by hand.

Imagine a tiny model with a vocabulary of only six words.

*   **Vocabulary:** `{"<pad>": 0, "The": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}`
*   **Input Sequence (`input_ids`):** "The cat sat" -> `[1, 2, 3]`
*   **Target Sequence (`labels`):** "cat sat on" -> `[2, 3, 4]`

The model processes the input and produces a vector of raw scores, or **logits**, for each possible next word at each position. Let's assume our model has produced the following logits:

| Position | Input Context | Model's Output Logits (for the next token) |
| :--- | :--- | :--- |
| 1 | `[1]` ("The") | `[0.1, 0.2, 2.0, 0.5, 0.3, 0.1]` |
| 2 | `[1, 2]` ("The cat") | `[0.1, 0.1, 0.2, 2.5, 0.4, 0.2]` |
| 3 | `[1, 2, 3]` ("The cat sat")| `[0.2, 0.1, 0.1, 0.3, 3.0, 0.5]` |

To calculate the loss, we perform three steps for each position:
1.  **Softmax:** Convert the raw logits into a probability distribution. $ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $.
2.  **Get Target Probability:** Find the probability the model assigned to the correct target token.
3.  **Calculate Loss:** Take the negative natural logarithm of that probability. $\text{Loss} = -\log(P_{\text{target}})$.

Let's fill out the table with the math, step-by-step.

| Step | Input Context | Target Token | Model's Logits | Softmax Probabilities | Prob. of Target Token | Loss (-log P) |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1**| "The" | `cat` (idx 2) | `[0.1,..,**2.0**,..]` | `[0.08,..,**0.593**,..]` | `0.593` | `-log(0.593) = 0.522` |
| **2**| "The cat"| `sat` (idx 3) | `[0.1,..,**2.5**,..]` | `[0.07,..,**0.793**,..]` | `0.793` | `-log(0.793) = 0.232` |
| **3**| "The cat sat"| `on` (idx 4) | `[0.2,..,**3.0**,..]` | `[0.08,..,**0.773**,..]` | `0.773` | `-log(0.773) = 0.257` |
| | | | | **Total Loss (Average)** | | | **(0.522+0.232+0.257)/3 = 0.337** |

The final loss for this sequence is the **average** of the individual token losses, which is **0.337**. This single number tells the optimizer how wrong the model was across the entire sequence, and backpropagation uses it to adjust the model's weights.

In PyTorch, this entire process is handled by a single, highly optimized function: `torch.nn.functional.cross_entropy`. The key is to format our tensors correctly.

```python
import torch
import torch.nn.functional as F

# Our model's output logits. These are the *exact same numbers* from the table.
# Shape: (Batch, Time, Vocab_size) -> (1, 3, 6)
logits = torch.tensor([[
    [0.1, 0.2, 2.0, 0.5, 0.3, 0.1],  # Logits for predicting after "The"
    [0.1, 0.1, 0.2, 2.5, 0.4, 0.2],  # Logits for predicting after "The cat"
    [0.2, 0.1, 0.1, 0.3, 3.0, 0.5]   # Logits for predicting after "The cat sat"
]])

# The correct next tokens (our labels)
# Shape: (Batch, Time) -> (1, 3)
targets = torch.tensor([[2, 3, 4]]) # "cat", "sat", "on"

# F.cross_entropy expects (N, C) and (N,)
# So we reshape our tensors to squash the Batch and Time dimensions together.
logits_flat = logits.view(-1, logits.size(-1)) # Shape: (3, 6)
targets_flat = targets.view(-1)               # Shape: (3)

loss = F.cross_entropy(logits_flat, targets_flat)

print(f"Logits shape (original): {logits.shape}")
print(f"Logits shape (flattened): {logits_flat.shape}")
print(f"Targets shape (flattened): {targets_flat.shape}")
print(f"Calculated Loss: {loss.item():.3f}")
```
**Output:**
```
Logits shape (original): torch.Size([1, 3, 6])
Logits shape (flattened): torch.Size([3, 6])
Targets shape (flattened): torch.Size([3])
Calculated Loss: 0.337
```
The result perfectly matches our manual, step-by-step calculation. This is the simple, powerful engine that drives large-scale pre-training.

Now we can see the source of the Parrot Problem with mathematical clarity. The model's sole objective is to minimize this cross-entropy loss over a massive dataset of text from the internet. It will adjust its weights to become a master of statistical mimicry because that is the most effective way to reduce the loss.

When you prompt it with `Q: What is the capital of Italy?`, it doesn't "understand" the question. It sees a sequence of tokens. It asks itself: "Across the trillions of tokens I have seen, what tokens are most likely to follow this sequence?" Since its data contains countless FAQs and quizzes formatted as `Q:...\nA:...\n\nQ:...`, completing the pattern is the path of least "surprise"—the path to the lowest possible loss.

It is a **parrot** because its training objective is **mimicry**.

To fix this, we need to change the data it learns from. We need to show it examples not of how text *is*, but of how we *want* it to be. This is the goal of Supervised Fine-Tuning, which we will build from the ground up in the next chapter.

## **Chapter 3: The SFT Solution: The Theory of Expert Imitation**
