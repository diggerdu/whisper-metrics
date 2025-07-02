"""
Whisper Metrics - Audio codec metrics calculation using whisper.cpp
"""

from .core import WhisperMetrics
from .model_manager import ModelManager
from .wer_calculator import WERCalculator

__version__ = "0.1.0"
__author__ = "Xingjian Du"
__email__ = "xingjian.du97@gmail.com"

__all__ = ["WhisperMetrics", "ModelManager", "WERCalculator"]