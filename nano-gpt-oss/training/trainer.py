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
            

            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                wandb.log({"train/loss": train_loss,
                           "val/loss": val_loss}, step=global_step)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "model/gotoss_best.pt")
                    artifact = wandb.Artifact("gptoss-model", type="model")
                    
                    artifact.add_file("model/gotoss_best.pt")
                    wandb.log_artifact(artifact)
                    print(f"✅ Saved new best model with val_loss={val_loss:.3f}")
        torch.save(model.state_dict(),"model/gotoss.pt")
        artifact = wandb.Artifact("gptoss-model", type="model")
        artifact.add_file("model/gotoss.pt")
        wandb.log_artifact(artifact, aliases=["latest"])

        torch.save([train_losses,val_losses,tokens_seen],"model/losses.pt")

        txt=generate_text(model,start_context)
        print(txt)
        wandb.log({"generated_text": wandb.Html(txt)}, step=global_step)
        clear_gpu_memory()
        
        
    return train_losses, val_losses, track_tokens_seen

def trainer(model,train_loader,val_loader,device):
    learning_rate = 3e-4        
    max_iters = 5         
    warmup_steps = 100    
    min_lr = 3e-5           
    eval_iters = 5
    eval_freq=150          
    
    
    torch.manual_seed(123)
    
    print(f"Using {device}")
   
    if os.path.exists('model/gptoss.pt'):
        model.load_state_dict(torch.load('model/gptoss.pt'))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    num_epochs=max_iters


    wandb.init(
    project="gptoss-VS-gpt2",
    name="gptoss-model",     
    group="model-comparison",  
    config={
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "eval_iters": eval_iters,
        "eval_freq": eval_freq,
        "device": device,
        "model_type": "gpt-oss"   
    }
)

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer,scheduler, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iters,
        start_context="a fast driver named Tim went for",
    )

    torch.save(model.state_dict(),"model/gotoss.pt")
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    wandb.log({"training_time_min": execution_time_minutes})
    wandb.finish()
    print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
    return train_losses,val_losses,tokens_seen
