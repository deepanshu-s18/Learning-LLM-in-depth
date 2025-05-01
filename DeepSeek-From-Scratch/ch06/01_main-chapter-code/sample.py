# sample.py
# This script loads a trained MiniDeepSeek model and generates text from a
# user-provided prompt.
# UPDATED for PyTorch 2.6+ security features.

import os
import json
import torch
import tiktoken
from model import MiniDeepSeek, ModelArgs
# ## NEW ##: Import the serialization module to handle the new security feature
from torch import serialization

# --- Configuration ---
out_dir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=pt_dtype) if 'cuda' in device else torch.no_grad()


## NEW ##: Updated this function for PyTorch 2.6+
def load_latest_checkpoint(directory: str):
    """Loads the latest checkpoint from the specified directory."""
    files = [f for f in os.listdir(directory) if f.startswith('ckpt_iter_') and f.endswith('.pt')]
    if not files:
        return None
    
    latest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    ckpt_path = os.path.join(directory, latest_file)
    print(f"Loading checkpoint: {ckpt_path}")
    
    # In PyTorch 2.6+, we must explicitly tell torch.load which classes are safe
    # to unpickle if we are loading more than just weights. Our checkpoint
    # contains the ModelArgs dataclass.
    with serialization.safe_globals([ModelArgs]):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
    return checkpoint

def main():
    # --- Load Model ---
    checkpoint = load_latest_checkpoint(out_dir)
    if checkpoint is None:
        print(f"No checkpoints found in '{out_dir}'. Please run train.py first.")
        return

    # The ModelArgs object is safely unpickled from the checkpoint
    model_args = checkpoint['model_args'] 
    model = MiniDeepSeek(model_args)
    
    # Load the trained model weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print("-" * 50)

    # --- Initialize Tokenizer ---
    enc = tiktoken.get_encoding("gpt2")

    # --- Interactive Generation Loop ---
    while True:
        try:
            start_text = input("Enter a prompt (or press Ctrl+C to exit): ")
            if not start_text:
                continue

            start_ids = enc.encode(start_text)
            x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            print(f"Prompt: '{start_text}'")
            print("Generating: ", end='', flush=True)

            kv_cache = None
            max_new_tokens = 100
            
            with torch.no_grad():
                with ctx:
                    for _ in range(max_new_tokens):
                        logits, kv_cache = model(x, past_kv_cache=kv_cache)
                        logits = logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1)
                        x = next_token.unsqueeze(0)
                        token_str = enc.decode([next_token.item()])
                        print(token_str, end='', flush=True)

            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == '__main__':
    main()