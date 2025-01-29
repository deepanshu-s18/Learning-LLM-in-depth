import os
import time
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from inference import generate_text

def calc_loss_batch(x, y, model, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

def calc_loss_loader(loader, model, device, num_batches=None):
    total = 0.0
    n     = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n

def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, eval_iter)
        vl = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    return tl, vl

def trainer(model, train_loader, val_loader, device):
    lr          = 5e-4
    min_lr      = 1e-5
    max_iters   = 5
    warmup      = 200
    eval_freq   = 100
    eval_iters  = 10

    if os.path.exists("model/gptoss.pt"):
        model.load_state_dict(torch.load("model/gptoss.pt"))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    s1  = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    s2  = CosineAnnealingLR(opt, T_max=max_iters - warmup, eta_min=min_lr)
    scheduler = SequentialLR(opt, [s1, s2], milestones=[warmup])

    best_val = float("inf")
    step     = 0
    t0       = time.time()

    for epoch in range(max_iters):
        model.train()
        for x, y in train_loader:
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            step += 1

            if step % eval_freq == 0:
                tl, vl = evaluate(model, train_loader, val_loader, device, eval_iters)
                print(f"Epoch {epoch+1} Step {step}: train={tl:.3f} val={vl:.3f}")
                if vl < best_val:
                    best_val = vl
                    os.makedirs("model", exist_ok=True)
                    torch.save(model.state_dict(), "model/best.pt")

        torch.save(model.state_dict(), "model/gptoss.pt")
        print(generate_text(model, "Once upon a time"))

    print(f"Done in {(time.time()-t0)/60:.1f} min")
