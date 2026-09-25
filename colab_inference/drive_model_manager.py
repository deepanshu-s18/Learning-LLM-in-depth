"""
drive_model_manager.py
======================
Python helper to easily save, checkpoint, and reload from-scratch models
and open-source weights directly from/to Google Drive in Google Colab.
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict, Any

class GoogleDriveModelHub:
    def __init__(self, base_dir: str = "/content/drive/MyDrive/LLM_Model_Hub"):
        self.base_dir = base_dir
        self.opensource_dir = os.path.join(base_dir, "open_source_models")
        self.scratch_dir = os.path.join(base_dir, "from_scratch_models")
        
        os.makedirs(self.opensource_dir, exist_ok=True)
        os.makedirs(self.scratch_dir, exist_ok=True)

    # -------------------------------------------------------------
    # From-Scratch Models Management
    # -------------------------------------------------------------
    def save_scratch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None
    ) -> str:
        """
        Saves weights, config, and training states of your custom from-scratch model directly to Google Drive.
        """
        save_path = os.path.join(self.scratch_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 1. State dict
        weights_file = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), weights_file)

        # 2. Config & metadata
        meta = {
            "model_name": model_name,
            "saved_at": datetime.utcnow().isoformat(),
            "step": step,
            "loss": loss,
            "config": config or {}
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # 3. Training Checkpoint
        if optimizer is not None:
            ckpt_file = os.path.join(save_path, "checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "loss": loss
            }, ckpt_file)

        print(f"✅ Model '{model_name}' saved to Google Drive: {save_path}")
        return save_path

    def load_scratch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> torch.nn.Module:
        """
        Loads saved weights for a from-scratch model from Google Drive.
        """
        weights_file = os.path.join(self.scratch_dir, model_name, "model.pt")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Model file not found at: {weights_file}")
            
        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"✅ Loaded weights for '{model_name}' on {device}")
        return model

    # -------------------------------------------------------------
    # Open-Source Weights Downloader
    # -------------------------------------------------------------
    def download_opensource_model(self, repo_id: str, hf_token: Optional[str] = None) -> str:
        """
        Downloads a full model repository from Hugging Face directly to Google Drive.
        """
        from huggingface_hub import snapshot_download

        folder_name = repo_id.replace("/", "--")
        dest_dir = os.path.join(self.opensource_dir, folder_name)
        print(f"⏳ Downloading {repo_id} to Google Drive: {dest_dir} ...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print(f"🎉 Model downloaded to: {dest_dir}")
        return dest_dir

    def list_inventory(self):
        """Prints all models currently stored in Google Drive."""
        print("=" * 60)
        print(f"📊 GOOGLE DRIVE MODEL INVENTORY: {self.base_dir}")
        print("=" * 60)
        
        print("\n🔹 FROM-SCRATCH MODELS:")
        if os.path.exists(self.scratch_dir):
            for item in os.listdir(self.scratch_dir):
                print(f"  • {item} -> {os.path.join(self.scratch_dir, item)}")
        
        print("\n🔹 OPEN-SOURCE MODELS:")
        if os.path.exists(self.opensource_dir):
            for item in os.listdir(self.opensource_dir):
                print(f"  • {item} -> {os.path.join(self.opensource_dir, item)}")
        print("=" * 60)
