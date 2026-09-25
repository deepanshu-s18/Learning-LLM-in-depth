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
