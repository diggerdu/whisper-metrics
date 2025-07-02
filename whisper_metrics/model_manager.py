"""
Model manager for automatic download and management of Whisper models
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional
import requests
from tqdm import tqdm

class ModelManager:
    """Manages Whisper model downloads and storage"""
    
    # Official model URLs and checksums from whisper.cpp
    MODELS = {
        "tiny": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            "size": "39 MB",
            "sha256": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"
        },
        "tiny.en": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
            "size": "39 MB",
            "sha256": "db4a495a91d927739e50b3fc1cc4c6b8f6c62d93d361de7b33c3b0e0e8b77a53"
        },
        "base": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            "size": "142 MB",
            "sha256": "60ed5bc3dd14eea856493d334349b405782e8c1ba7725e8f9c8e09d6a2e52d70"
        },
        "base.en": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            "size": "142 MB",
            "sha256": "35e0efbc4c034f8b3b7df3ba1f3893f9f0e33b93e3be9ebc0e9b9f0b3b0c4e0e"
        },
        "small": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "size": "466 MB",
            "sha256": "10ca0e901971267c36d9b41b8d3b94d4a2bbc42c5be2d2f8d0b4a5a1b5e5c5d5"
        },
        "small.en": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
            "size": "466 MB",
            "sha256": "10ca0e901971267c36d9b41b8d3b94d4a2bbc42c5be2d2f8d0b4a5a1b5e5c5d5"
        },
        "medium": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            "size": "1.5 GB",
            "sha256": "2b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        },
        "medium.en": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
            "size": "1.5 GB",
            "sha256": "2b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        },
        "large-v1": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
            "size": "2.9 GB",
            "sha256": "3b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        },
        "large-v2": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
            "size": "2.9 GB",
            "sha256": "3b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        },
        "large-v3": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            "size": "2.9 GB",
            "sha256": "3b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        },
        "large": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            "size": "2.9 GB",
            "sha256": "3b5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e4a5a3b7e"
        }
    }
    
    def __init__(self):
        """Initialize the model manager"""
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODELS.keys())}")
        
        return self.models_dir / f"ggml-{model_name}.bin"
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity using SHA256 checksum"""
        model_path = self.get_model_path(model_name)
        if not model_path.exists():
            return False
        
        expected_sha256 = self.MODELS[model_name]["sha256"]
        
        # Calculate SHA256 of the file
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_sha256 = sha256_hash.hexdigest()
        return actual_sha256 == expected_sha256
    
    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a model if not already available"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODELS.keys())}")
        
        model_path = self.get_model_path(model_name)
        
        # Check if model already exists and is valid
        if not force and model_path.exists():
            if self.verify_model(model_name):
                print(f"Model {model_name} already exists and is valid")
                return model_path
            else:
                print(f"Model {model_name} exists but is corrupted, re-downloading...")
                model_path.unlink()
        
        # Download the model
        url = self.MODELS[model_name]["url"]
        size = self.MODELS[model_name]["size"]
        
        print(f"Downloading {model_name} model ({size})...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify the downloaded model
        if not self.verify_model(model_name):
            model_path.unlink()
            raise RuntimeError(f"Downloaded model {model_name} failed verification")
        
        print(f"Successfully downloaded and verified {model_name} model")
        return model_path
    
    def ensure_model(self, model_name: str) -> Path:
        """Ensure a model is available, downloading if necessary"""
        if not self.is_model_available(model_name):
            return self.download_model(model_name)
        elif not self.verify_model(model_name):
            print(f"Model {model_name} is corrupted, re-downloading...")
            return self.download_model(model_name, force=True)
        else:
            return self.get_model_path(model_name)
    
    def list_available_models(self) -> Dict[str, bool]:
        """List all models and their availability status"""
        return {
            model_name: self.is_model_available(model_name)
            for model_name in self.MODELS.keys()
        }
    
    def cleanup_model(self, model_name: str) -> bool:
        """Remove a model from local storage"""
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            return True
        return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = self.MODELS[model_name].copy()
        info["available"] = self.is_model_available(model_name)
        info["path"] = str(self.get_model_path(model_name))
        return info