# **Title: Direct Preference Optimization (DPO): From Preference to Policy**

## **Chapter 1: The Alignment Problem: Judging is Easier than Creating**

A pre-trained Large Language Model (LLM) is a modern marvel. Trained on a vast expanse of the internet, it's a master of language, grammar, and knowledge. However, out of the box, it's not a helpful assistant; it's a **statistical parrot**. Its only goal is to predict the next word in a sequence, which can lead to some very unhelpful behavior.

For instance, if you ask a raw base model a question, you might get this:

**Your Prompt:**
```
What is the primary cause of Earth's seasons?
```

**The Base Model's Likely Response:**
```
What is the primary cause of Earth's seasons?
A) The Earth's distance from the sun.
B) The tilt of the Earth's axis.
C) The speed of the Earth's rotation.
D) Ocean currents and wind patterns.
```

The model didn't answer your question. It continued your text by turning it into a multiple-choice quiz. Why? Because it has seen countless quizzes online and, from a statistical standpoint, this is a highly probable pattern. It has no concept of your *intent*.

#### The First Solution: Supervised Fine-Tuning (SFT)

The first step in fixing this is **Supervised Fine-Tuning (SFT)**. The core idea is simple: we teach the model to be a helpful assistant by showing it thousands of high-quality examples. We curate a dataset of `(prompt, ideal_response)` pairs and train the model to imitate the ideal responses.

This process is incredibly effective. It takes the raw, aimless parrot and turns it into a capable apprentice that understands conversational structure and follows instructions.

| Characteristic | Pre-trained Base Model (The Parrot) | SFT Model (The Apprentice) |
| :--- | :--- | :--- |
| **Training Goal** | Predict the next word in any text. | Imitate expert-written responses to specific prompts. |
| **Training Data**| Unstructured internet text (books, websites, code).| Curated `(prompt, ideal_response)` pairs. |
| **Behavior** | Completes text patterns; no sense of user intent. | Follows instructions; adopts a helpful persona. |
| **Key Weakness** | Doesn't know how to be a helpful assistant. | Assumes there is only one "perfect" answer for every prompt. |

#### The SFT Limitation: The "Shades of Gray" Problem

SFT is a huge leap forward, but it has a fundamental weakness: it treats alignment as a black-and-white problem. The provided response in the dataset is considered 100% correct, and any deviation is implicitly wrong.

The real world, however, is full of nuance. Often, there isn't one "perfect" answer. Consider these two AI-generated summaries of an article:

*   **Response A:** "The article discusses climate change, focusing on rising sea levels and CO2 emissions. It mentions policy solutions." (Factually correct, but basic).
*   **Response B:** "The article provides a detailed analysis of climate change, attributing rising sea levels primarily to thermal expansion and glacial melt. It contrasts market-based policy solutions, like carbon taxes, with regulatory approaches." (More detailed, nuanced, and helpful).

As a human, you can instantly state a preference: **B is better than A**. SFT has no way to learn this relative judgment. It can only imitate.

This brings us to the core economic and practical insight that powers the next generation of alignment techniques:

> **It is exponentially easier for a human to *judge* which of two responses is better than it is to *create* a single perfect response from scratch.**

Creating a "perfect" response for an SFT dataset is time-consuming and expensive. Simply choosing `A` or `B` is fast and cheap. If we could build an algorithm that learns directly from these simple preferences, we could align our models more efficiently and effectively.

That algorithm is **Direct Preference Optimization (DPO)**.

By the end of this tutorial, you will understand every single component of the DPO formula. It may look intimidating now, but we are going to build it together, piece by piece, starting from this simple idea of human preference.

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right] $$

Our journey begins in the next chapter, where we will translate the simple human judgment "B is better than A" into our first piece of the mathematical puzzle.


## **Chapter 2: From Scores to Probabilities: The Bradley-Terry Model**

In the last chapter, we established that human preference data—simply judging "A is better than B"—is a powerful and efficient way to align LLMs. But how do we take a qualitative judgment like "Response B is better than Response A" and turn it into something a computer can understand and learn from? We need a mathematical model.

#### The Intuition: Hidden Quality Scores

The most intuitive way to model a preference is to assume that each option possesses an underlying, hidden **quality score**. Let's call this score **$r$**. When a human expresses a preference, they are implicitly stating that the chosen option has a higher quality score than the rejected one.

Let's revisit our example from Chapter 1:

*   **Response B:** "The article provides a detailed analysis of climate change, attributing rising sea levels primarily to thermal expansion and glacial melt. It contrasts market-based policy solutions, like carbon taxes, with regulatory approaches." (The preferred one)
*   **Response A:** "The article discusses climate change, focusing on rising sea levels and CO2 emissions. It mentions policy solutions." (The rejected one)

If a human judges **B > A**, we can assume:
*   $r_{\text{B}}$ (quality score for Response B) > $r_{\text{A}}$ (quality score for Response A)

For our tutorial, let's assign some hypothetical scores to make this concrete:
*   $r_{\text{winner}}$ (for Response B): **2.5**
*   $r_{\text{loser}}$ (for Response A): **0.8**

Our goal is to build a system that, given these scores, can predict the human's preference with a probability.

#### Formalizing Preferences: The Bradley-Terry Model

A simple yet powerful mathematical framework for modeling pairwise comparisons like this is the **Bradley-Terry model**. It states that the probability of one item (the winner) being preferred over another (the loser) is a function of the *difference* in their underlying quality scores.

The formula is:

$$ P(\text{winner} \succ \text{loser}) = \sigma(r_{\text{winner}} - r_{\text{loser}}) $$

Let's break down this formula:

*   **$P(\text{winner} \succ \text{loser})$:** This is the probability that the winner is indeed preferred over the loser, according to our model.
*   **$r_{\text{winner}}$ and $r_{\text{loser}}$:** These are the quality scores we just discussed.
*   **$\sigma$:** This is the **sigmoid function**. It's a fundamental tool in machine learning for converting any real number into a probability between 0 and 1.
    *   The sigmoid function is defined as: $\sigma(x) = \frac{1}{1 + e^{-x}}$
    *   It squashes values:
        *   Large positive `x` (big score difference) -> `σ(x)` approaches 1.0 (high probability)
        *   `x = 0` (no score difference) -> `σ(x)` is 0.5 (50/50 probability)
        *   Large negative `x` (loser's score is much higher) -> `σ(x)` approaches 0.0 (low probability)

#### Step-by-Step Calculation with Our Example

Let's apply the Bradley-Terry model to our concrete example:

1.  **Calculate the score difference ($\Delta r$):**
    $\Delta r = r_{\text{winner}} - r_{\text{loser}} = 2.5 - 0.8 = 1.7$

2.  **Apply the sigmoid function:**
    $P(\text{winner} \succ \text{loser}) = \sigma(1.7) = \frac{1}{1 + e^{-1.7}} = \frac{1}{1 + 0.1827} \approx 0.845$

Our model predicts an **84.5% probability** that Response B would be preferred over Response A. This makes intuitive sense: a larger positive difference in quality scores should correspond to a higher probability of preference.

The sigmoid function's behavior for different score differences is crucial to grasp:

| Score Difference ($\Delta r$) | Sigmoid($\Delta r$) | Interpretation |
| :--- | :--- | :--- |
| Large Positive (e.g., 5.0) | ~0.993 | Almost certain winner is preferred. |
| **Our Example (1.7)** | **~0.845** | **Confident winner is preferred.** |
| Zero (e.g., 0.0) | 0.500 | Completely uncertain; a 50/50 toss-up. |
| Negative (e.g., -1.7) | ~0.155 | Confident *loser* is preferred (our scores are "wrong" for the human judgment). |
| Large Negative (e.g., -5.0)| ~0.007 | Almost certain loser is preferred. |

We have successfully taken a simple human preference and converted it into a quantifiable probability. This `P(winner > loser)` is the first critical piece of our DPO puzzle. In the next chapter, we'll see how to turn this probability into a "loss" that our LLM can learn from.

## **Chapter 3: From Probability to a Loss Function: Negative Log-Likelihood**

In the last chapter, we successfully used the Bradley-Terry model to convert a pair of abstract "quality scores" into a probability. We called this `P(winner ≻ loser)`, and for our example, we calculated its value to be `0.845`.

This is a great first step, but a probability is just a prediction. To train a machine learning model, we need an **error signal**—a single number that tells the model how wrong its prediction was. This error signal is called a **loss function**, and the model's entire goal during training is to minimize this value.

The standard, time-tested method for converting a probability into a loss is to calculate the **Negative Log-Likelihood (NLL)**.

The formula is beautifully simple:

$$ \mathcal{L} = -\log \left( P(\text{winner} \succ \text{loser}) \right) $$

Now, we can combine this with the Bradley-Terry formula from Chapter 2 to create the complete loss function for our abstract scores:

$$ \mathcal{L} = -\log(\sigma(r_{\text{winner}} - r_{\text{loser}})) $$

#### A Quick Detour: Why Use the Negative Logarithm?

Why not use a simpler function, like `Loss = 1 - P`? The negative logarithm has a specific, powerful property that makes it perfect for training models.

Let's compare the two:

| Model's Probability `P` | Simpler Loss (`1 - P`) | **NLL Loss (`-log(P)`)** |
| :--- | :--- | :--- |
| 0.99 (Very Confident & Correct) | 0.01 | **0.01** |
| 0.50 (Uncertain) | 0.50 | **0.69** |
| 0.10 (Confident & Wrong) | 0.90 | **2.30** |
| 0.01 (Very Confident & Wrong) | 0.99 | **4.61** |
| 0.001 (Extremely Confident & Wrong) | 0.999 | **6.91** |

Notice the difference in the last few rows. With the simpler `1 - P` loss, the penalty for being wrong (`0.90`) and *very* wrong (`0.99`) is quite similar.

The **Negative Log-Likelihood**, however, *explodes* towards infinity as the probability approaches zero. It **aggressively penalizes** the model for being both confident and wrong. This creates a much stronger "error signal" (a steeper gradient) that forces the model to correct its biggest mistakes, leading to faster and more stable training. In information theory, `-log(P)` is a measure of "surprisal" — a highly improbable event is very surprising and contains a lot of information for the model to learn from.

#### Step-by-Step Calculation with Our Example

Let's calculate the loss for our ongoing example. In Chapter 2, we found that `P(winner ≻ loser)` was `0.845`.

1.  **Take the natural logarithm:**
    $\log(0.845) \approx -0.168$

2.  **Negate the result:**
    $\mathcal{L} = -(-0.168) = \mathbf{0.168}$

This single number, `0.168`, is our error signal for this specific preference pair. The job of a machine learning optimizer (like Adam) is to adjust the underlying parameters (in our case, the scores `r`) to push this loss value as close to zero as possible.

We have now built a complete, self-contained **preference modeling engine**. It takes two abstract scores, `r_winner` and `r_loser`, and produces a single, trainable loss value.

We are now ready to leave the world of abstract scores behind and connect this engine to a real Large Language Model. The critical question we must answer next is: how do we calculate these quality scores, `r`, for a sequence of text generated by an LLM? That is the subject of the next part of our journey.

## **Chapter 4: A Naive Reward for LLMs: Using Log-Probabilities**

In the last three chapters, we built a powerful "preference engine." It takes two numerical quality scores, `r_winner` and `r_loser`, and computes a loss that can be used for training.

This engine is abstract and general-purpose. Now, we must connect it to our specific problem: aligning a Large Language Model. The critical question we need to answer is:

> How do we define a numerical "quality score" (`r`) for a sequence of text generated by an LLM?

#### The Intuition: Confidence as a Proxy for Quality

Let's start with the most direct and intuitive idea. A "good" response should be one that our trainable language model is highly confident in generating. Conversely, a "bad" response should be one the model finds unlikely. If we can encourage the model to become more confident in the responses humans prefer, we should be able to steer its behavior in the right direction.

How do we measure an LLM's confidence in a given sequence of text? We use its **sequence log-probability**.

This leads us to our first, naive hypothesis for the reward function.

**Naive Reward Hypothesis:** The quality score `r` of a response `y` given a prompt `x` is the total log-probability of generating that response, as calculated by our trainable policy model, $\pi_{\theta}$.

#### The Formal Math

This hypothesis translates into the following mathematical formula for our score, `r`:

$$ r(x, y) = \log \pi_{\theta}(y|x) $$

Let's break this down. The probability of an entire sequence `y` (which consists of tokens $y_1, y_2, ..., y_N$) is the product of the probabilities of generating each token one by one:

$$ \pi_{\theta}(y|x) = \pi_{\theta}(y_1|x) \times \pi_{\theta}(y_2|x, y_1) \times \dots \times \pi_{\theta}(y_N|x, y_{<N}) $$

Multiplying many small probabilities together is numerically unstable in a computer. To fix this, we work in log space, where products become simple sums:

$$ r(x, y) = \log \pi_{\theta}(y|x) = \sum_{t=1}^{N} \log \pi_{\theta}(y_t | x, y_{<t}) $$

*   **What this means:** The total score of a response is the **sum** of the log-probabilities of each of its individual tokens.

Let's make this concrete with a simple example:

*   **Prompt `x`:** "The capital of France"
*   **Response `y`:** " is Paris"

The model calculates the score in two steps:
1.  Given the context "The capital of France", it calculates the log-probability of the next token being " is". Let's say it's `-0.2`.
2.  Given the context "The capital of France is", it calculates the log-probability of the next token being "Paris". Let's say it's `-0.3`.

The final score for the response " is Paris" would be the sum:
$r(x, y) = (-0.2) + (-0.3) = \mathbf{-0.5}$

This approach seems perfectly logical. To make the model prefer a winning response `y_w` over a losing one `y_l`, we just need to train it to produce a higher log-probability (a less negative score) for `y_w`.

We now have our first concrete proposal for an LLM reward function. But to use it, we need to know how to actually compute this value in code. In the next chapter, we will do exactly that, writing a PyTorch function from scratch to calculate the sequence log-probability for any given response.

## **Chapter 5: Code Deep Dive: Calculating Sequence Log-Probabilities**

In the last chapter, we defined our first reward function for an LLM: the sequence log-probability. Now, we need to translate that mathematical concept into working PyTorch code. Our goal is to create a single, robust function that can calculate this score for any given prompt and response.

Before we build our function, let's have a quick refresher on how a language model generates a single token. This will help us understand where the probabilities we need come from.

#### A Quick Refresher: From Logits to Probabilities

When you give a language model a sequence of input tokens, it performs a single "forward pass" and produces a final tensor called **logits**.

1.  **Input:** A sequence of token IDs, e.g., `input_ids = [1, 2, 3, 4]` ("The capital of France").
2.  **Model Forward Pass:** The model processes these IDs and outputs `logits`.
3.  **Logits:** This is a large tensor of raw, unnormalized scores. Its shape is typically `(batch_size, sequence_length, vocab_size)`. For our input, the shape would be `(1, 4, 50257)` if we use a GPT-2 sized vocabulary. The key part is the last logit vector, `logits[0, -1, :]`, which contains a score for every single word in the vocabulary for being the *next* token.
4.  **Probabilities:** To turn these raw scores into probabilities, we apply the **softmax function**.

```python
import torch
import torch.nn.functional as F

# A mock logit vector for the last token. High score for "is".
# Shape: (vocab_size)
mock_logits = torch.tensor([0.1, 3.0, 0.5, 0.2]) # Vocab: {"The":0, "is":1, "Paris":2, "Lyon":3}

# Convert logits to probabilities
probabilities = F.softmax(mock_logits, dim=-1)
print(f"Probabilities: {probabilities.numpy()}")
# Output: Probabilities: [0.045... 0.819... 0.061... 0.049...]
```
During generation, the model would *sample* from this probability distribution to pick the next token. But for DPO, we don't want to sample. We already have the response. We just need to find the probability the model assigned to the specific tokens that were *actually in that response*.

#### The `get_sequence_log_probs` Function

Now we're ready to build our core function. It will take a batch of prompts and responses and return the total log-probability for each response.

```python
def get_sequence_log_probs(model, prompt_tokens, response_tokens):
