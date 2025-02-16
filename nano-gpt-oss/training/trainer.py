import torch
from inference import generate_text
import time,os,gc
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from tqdm.notebook import tqdm

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def calcc(input_batch, target_batch, model,device):
    total_loss=0
    for i in range(len(input_batch)):
        inp = input_batch[i].to(device, non_blocking=True)   
        logits = model(inp)
        del inp
        tgt = target_batch[i].to(device, non_blocking=True)
        loss= torch.nn.functional.cross_entropy(logits,tgt)
        total_loss += loss
        del tgt, logits, loss
        
    clear_gpu_memory()  
    return total_loss/len(input_batch)


def calc_loss_batch(input_batch, target_batch, model, device):
    loss=calcc(input_batch,target_batch,model,device)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            del loss
            
        else:
            break
    clear_gpu_memory()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss







def train_model(model, train_loader, val_loader, optimizer,scheduler, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        artifact = wandb.Artifact("gptoss-model", type="model")
        for input_batch,target_batch in tqdm(train_loader):

            loss= calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            wandb.log({"train/step_loss": loss.item(), 
                       "tokens_seen": tokens_seen, 
                       "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]
                       }, step=global_step)
