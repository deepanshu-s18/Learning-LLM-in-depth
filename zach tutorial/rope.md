# **Title: Rotary Positional Encoding (RoPE): A Deep Dive**

## **Introduction: Give me 40 minutes, and I will make RoPE click forever.**

You have likely heard that modern Large Language Models like Llama, PaLM, and GPT-NeoX have abandoned traditional positional embeddings. Instead, they use a more powerful and elegant method: **Rotary Positional Encoding**, or **RoPE**.

What if I told you the entire "magic" behind this technique is just high-school trigonometry? What if the core of this powerful idea is captured in this single matrix operation?

$$
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) \\ \sin(m\theta_0) & \cos(m\theta_0) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
$$

This is the 2D rotation matrix. It's the entire secret. By the end of this tutorial, this formula will not only make sense, but you will understand how it's generalized to high dimensions and why it's the key to unlocking **relative positional information** in Transformers.

#### **Our Promise**

Our promise is simple: in the next 40 minutes, you will understand RoPE from first principles to practical implementation. You will understand not only **what** the algorithm is, but **why** it works so effectively. By the end, you will understand this code:

```python
def apply_rotary_pos_emb(x: torch.Tensor, rope_emb: torch.Tensor):
    seq_len = x.shape[1]
    rope_emb_sliced = rope_emb[:seq_len, :].unsqueeze(0).unsqueeze(2)
    cos_emb = rope_emb_sliced.cos()
    sin_emb = rope_emb_sliced.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)
