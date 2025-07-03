"""
Whisper Metrics - Audio codec metrics calculation using whisper.cpp
"""

from .core import WhisperMetrics
from .model_manager import ModelManager
from .wer_calculator import WERCalculator

__version__ = "0.1.0"
__author__ = "Xingjian Du"
__email__ = "xingjian.du97@gmail.com"

def is_cuda_available():
    """Check if CUDA acceleration is available"""
    try:
        from pathlib import Path
        bin_dir = Path(__file__).parent / "bin"
        # Check if CUDA library exists
        for f in bin_dir.iterdir():
            if 'cuda' in f.name.lower() and f.name.endswith('.so'):
                return True
        return False
    except:
        return False

__all__ = ["WhisperMetrics", "ModelManager", "WERCalculator", "is_cuda_available"]