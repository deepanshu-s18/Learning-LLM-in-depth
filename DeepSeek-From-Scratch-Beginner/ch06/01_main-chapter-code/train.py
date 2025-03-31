import os, json, time, math
import numpy as np
import torch
from model import MiniDeepSeek, ModelArgs

device     = "cuda" if torch.cuda.is_available() else "cpu"
data_dir   = "data/tinystories_tokenized"
out_dir    = "out"
max_iters  = 5000
batch_size = 32
block_size = 128
lr         = 3e-4

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([torch.from_numpy(data[i: i+block_size].astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i+1: i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def cosine_lr(it):
    warmup = 100
    if it < warmup:
        return lr * it / warmup
    decay = (it - warmup) / (max_iters - warmup)
    return (lr / 10) + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * decay))

if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data_dir, "meta.json")) as f:
        vocab_size = json.load(f)["vocab_size"]

    args  = ModelArgs(d_model=768, n_layers=12, num_heads=12, d_latent=192,
                      d_rope=64, vocab_size=vocab_size, max_seq_len=block_size)
    model = MiniDeepSeek(args).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    for step in range(max_iters):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step)
        model.train()
        x, y   = get_batch("train")
        out    = model(x, targets=y)
        loss   = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f} lr={cosine_lr(step):.6f}")
    print("Done training.")
