# **Give Me 1 Hour, You Will Build and Train a Neural Network From Scratch**

## Intro

You've seen PyTorch code. Maybe in a research paper or a tutorial. You saw `model.train()`, `loss.backward()`, and `optimizer.step()`.

You wondered what they actually *do*.

Maybe you tried to read the documentation. Got hit with a wall of classes. `nn.Module`, `torch.optim`, computation graphs. It felt like opaque, magical black boxes.

**Here's the truth: They're not black boxes. They are elegant wrappers for the linear algebra and calculus you already know.**

The entire deep learning training process, from the simplest model to a giant LLM, is just a five-step algorithm. That's it. Everything else is just an implementation detail.

- The `nn.Linear` layer? It's just `X @ W + b`.
- The loss function? Just a formula to measure error, like `torch.mean((y - y_hat)**2)`.
- The magical `loss.backward()`? It's just your old friend, the chain rule (backpropagation), automated.
- The `optimizer.step()`? It's just the gradient descent update rule: `W -= learning_rate * W.grad`.

**In this guide, we'll implement the raw math first, then use PyTorch's professional tools.**

You'll see *why* the tools exist by first doing the work manually. Once you've implemented the raw math, the professional code becomes obvious—just a clean, powerful way to organize what you've already built.

**And it all boils down to these 5 steps.**

Master this five-step loop, and you can understand the training code for any model, from a simple regression to a state-of-the-art Large Language Model.

| Step | Concept (Math) | From Scratch (Raw Tensors) | Professional (`torch.nn`) |
| :--- | :--- | :--- | :--- |
| **1. Prediction** | Make a prediction: <br> `ŷ = f(X, θ)` | `y_hat = X @ W + b` | `y_hat = model(X)` |
| **2. Loss Calc** | Quantify the error: <br> `L = Loss(ŷ, y)` | `loss = torch.mean((y_hat - y)**2)` | `loss = criterion(y_hat, y)` |
| **3. Gradient Calc** | Find the slope of the loss: <br> `∇_θ L` | `loss.backward()` | `loss.backward()` |
| **4. Param Update** | Step down the slope: <br> `θ_t+1 = θ_t - η * ∇_θ L` | `W -= lr * W.grad` | `optimizer.step()` |
| **5. Gradient Reset** | Reset for the next loop. | `W.grad.zero_()` | `optimizer.zero_grad()` |

Even better—this loop is universal. The `model` in Step 1 might become a massive Transformer, and the `optimizer` in Step 4 might be a fancy one like Adam, but the fundamental five-step logic **never changes**.

This isn't an analogy. The `nn.Linear` layer used in LLMs is the *exact same* `X @ W + b` operation you will build in Part 2. The Feed-Forward Network inside a Transformer is the *exact same* stack of layers you will build in Part 4. You are learning the literal, fundamental building blocks of modern AI.

Give me one hour. We'll build your understanding from the ground up. Raw math, real code, and a clear path to the professional tools.

Ready? Let's begin.

## Part 1: The Core Data Structure - `torch.Tensor`

Welcome to PyTorch! Everything we do—from loading data to defining complex neural networks—will be built on one core object: the `torch.Tensor`.

**The Analogy:** Think of a `Tensor` as the fundamental building block, the basic "noun" of the PyTorch language. It's a multi-dimensional array, very similar to a NumPy array, but with special powers for deep learning that we will unlock step-by-step.

Our goal in this section is to master this single, crucial object: how to create it and how to understand its properties.

### 1.1. The Three Common Patterns for Creating Tensors

There are many ways to create a tensor, but they generally fall into three common patterns you'll see everywhere.

#### **Pattern 1: Direct Creation from Existing Data**
This is the most straightforward. You have a Python list of numbers, and you want to convert it into a PyTorch tensor using `torch.tensor()`.

```python
import torch

# Input: A standard Python list
data = [[1, 2, 3], [4, 5, 6]]
my_tensor = torch.tensor(data)

print(my_tensor)
```
**Output:**
```
tensor([[1, 2, 3],
        [4, 5, 6]])
```

---

#### **Pattern 2: Creation from a Desired Shape**
This is extremely common when building models. You don't know the exact values for your weights yet, but you know the dimensions (shape) you need.

*   `torch.ones()` or `torch.zeros()`: Fills the tensor with `1`s or `0`s.
*   `torch.randn()`: Fills the tensor with random numbers from a standard normal distribution. **This is the standard way to initialize a model's weights.**

```python
# Input: A shape tuple (2 rows, 3 columns)
shape = (2, 3)

# Create tensors with this shape
ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)

print(f"Ones Tensor:\n {ones}\n")
print(f"Zeros Tensor:\n {zeros}\n")
print(f"Random Tensor:\n {random}")
```
**Output:**
```
Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])

Random Tensor:
 tensor([[ 0.3815, -0.9388,  1.6793],
        [-0.3421,  0.5898,  0.3609]])
```

---

#### **Pattern 3: Creation by Mimicking Another Tensor's Properties**
Often, you have an existing tensor and you need to create a new one that has the *exact same shape, data type, and is on the same device*. Instead of manually copying these properties, PyTorch provides handy `_like` functions.

```python
# Input: A 'template' tensor
template = torch.tensor([[1, 2], [3, 4]])

# Create new tensors with the same properties as the template
ones_like = torch.ones_like(template)
rand_like = torch.randn_like(template, dtype=torch.float) # dtype can be overridden

print(f"Template Tensor:\n {template}\n")
print(f"Ones_like Tensor:\n {ones_like}\n")
print(f"Randn_like Tensor:\n {rand_like}")
```
**Output:**
```
Template Tensor:
 tensor([[1, 2],
        [3, 4]])

Ones_like Tensor:
 tensor([[1, 1],
        [1, 1]])

Randn_like Tensor:
 tensor([[-1.1713, -0.2032],
        [ 0.3391, -0.8267]])
```

### 1.2. What's Inside a Tensor? Shape, Type, and Device

Every tensor has attributes that describe its metadata. The three you will use constantly are `.shape`, `.dtype`, and `.device`.

```python
# Let's create a tensor to inspect
tensor = torch.randn(2, 3)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
**Output:**
```
Shape of tensor: torch.Size([2, 3])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

Let's break these down:
*   **`.shape`**: This is a tuple that describes the dimensions of the tensor. `torch.Size([2, 3])` tells us it's a 2D tensor with 2 rows and 3 columns. This is the most important attribute for debugging your models. Mismatched shapes are the #1 source of errors in PyTorch.
*   **`.device`**: This tells you where the tensor's data is physically stored. By default, it's on the `cpu`. If you have a compatible GPU, you can move it there (`.to("cuda")`) for massive speedups.
*   **`.dtype`**: This describes the data type of the numbers inside the tensor. Notice it defaulted to `torch.float32`. This is not an accident, and it's critically important.
