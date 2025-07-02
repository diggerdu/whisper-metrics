"""
Main API for Whisper Metrics package
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Any

from .model_manager import ModelManager
from .whisper_binding import WhisperBinding
from .wer_calculator import WERCalculator
from .audio_utils import load_audio, generate_noise, save_audio_wav

class WhisperMetrics:
    """
    Main class for calculating audio codec metrics using Whisper transcription
    """
    
    def __init__(self, 
                 model: str = "base",
                 auto_download: bool = True,
                 wer_config: Optional[Dict[str, Any]] = None):
        """
        Initialize WhisperMetrics
        
        Args:
            model: Whisper model name (tiny, base, small, medium, large)
            auto_download: Automatically download model if not available
            wer_config: Configuration for WER calculator
        """
        self.model_name = model
        self.auto_download = auto_download
        
        # Initialize components
        self.model_manager = ModelManager()
        self.whisper_binding = WhisperBinding()
        
        # Initialize WER calculator with config
        wer_config = wer_config or {}
        self.wer_calculator = WERCalculator(**wer_config)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        if self.auto_download:
            model_path = self.model_manager.ensure_model(self.model_name)
        else:
            model_path = self.model_manager.get_model_path(self.model_name)
            if not model_path.exists():
                raise FileNotFoundError(f"Model {self.model_name} not found. Set auto_download=True to download automatically.")
        
        success = self.whisper_binding.load_model(model_path)
        if not success:
            raise RuntimeError(f"Failed to load model {self.model_name}")
    
    def transcribe_audio(self, 
                        audio_path: Union[str, Path],
                        with_timestamps: bool = False) -> Union[str, List[Dict]]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            with_timestamps: Return timestamps with transcription
            
        Returns:
            Transcribed text or list of segments with timestamps
        """
        # Load audio
        audio_data, sample_rate = load_audio(audio_path, target_sr=16000, mono=True)
        
        # Transcribe
        if with_timestamps:
            return self.whisper_binding.transcribe_with_timestamps(audio_data)
        else:
            return self.whisper_binding.transcribe(audio_data)
    
    def calculate_wer_with_reference(self, 
                                   reference_text: str,
                                   generated_audio_path: Union[str, Path],
                                   return_all_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate WER between reference text and generated audio transcription
        
        Args:
            reference_text: Reference text transcription
            generated_audio_path: Path to generated audio file
            return_all_metrics: Return all metrics (WER, MER, WIL, WIP, CER)
            
        Returns:
            WER score or dictionary of all metrics
        """
        # Transcribe generated audio
        generated_text = self.transcribe_audio(generated_audio_path)
        
        # Calculate metrics
        if return_all_metrics:
            return self.wer_calculator.calculate_all_metrics(reference_text, generated_text)
        else:
            return self.wer_calculator.calculate_wer(reference_text, generated_text)
    
    def calculate_wer_audio_to_audio(self, 
                                   reference_audio_path: Union[str, Path],
                                   generated_audio_path: Union[str, Path],
                                   return_all_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate WER between transcriptions of reference and generated audio
        
        Args:
            reference_audio_path: Path to reference audio file
            generated_audio_path: Path to generated audio file
            return_all_metrics: Return all metrics (WER, MER, WIL, WIP, CER)
            
        Returns:
            WER score or dictionary of all metrics
        """
        # Transcribe both audio files
        reference_text = self.transcribe_audio(reference_audio_path)
        generated_text = self.transcribe_audio(generated_audio_path)
        
        # Calculate metrics
        if return_all_metrics:
            return self.wer_calculator.calculate_all_metrics(reference_text, generated_text)
        else:
            return self.wer_calculator.calculate_wer(reference_text, generated_text)
    
    def batch_calculate_wer_with_reference(self, 
                                         reference_texts: List[str],
                                         generated_audio_paths: List[Union[str, Path]],
                                         return_all_metrics: bool = False) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate WER for multiple audio files with reference texts
        
        Args:
            reference_texts: List of reference text transcriptions
            generated_audio_paths: List of paths to generated audio files
            return_all_metrics: Return all metrics for each file
            
        Returns:
            List of WER scores or list of metric dictionaries
        """
        if len(reference_texts) != len(generated_audio_paths):
            raise ValueError("Number of reference texts must match number of generated audio files")
        
        results = []
        for ref_text, audio_path in zip(reference_texts, generated_audio_paths):
            result = self.calculate_wer_with_reference(ref_text, audio_path, return_all_metrics)
            results.append(result)
        
        return results
    
    def batch_calculate_wer_audio_to_audio(self, 
                                         reference_audio_paths: List[Union[str, Path]],
                                         generated_audio_paths: List[Union[str, Path]],
                                         return_all_metrics: bool = False) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate WER for multiple pairs of audio files
        
        Args:
            reference_audio_paths: List of paths to reference audio files
            generated_audio_paths: List of paths to generated audio files
            return_all_metrics: Return all metrics for each pair
            
        Returns:
            List of WER scores or list of metric dictionaries
        """
        if len(reference_audio_paths) != len(generated_audio_paths):
            raise ValueError("Number of reference and generated audio files must match")
        
        results = []
        for ref_path, gen_path in zip(reference_audio_paths, generated_audio_paths):
            result = self.calculate_wer_audio_to_audio(ref_path, gen_path, return_all_metrics)
            results.append(result)
        
        return results
    
    def evaluate_with_detailed_output(self, 
                                    reference_text: str,
                                    generated_audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Evaluate with detailed output including transcriptions and all metrics
        
        Args:
            reference_text: Reference text transcription
            generated_audio_path: Path to generated audio file
            
        Returns:
            Dictionary with detailed evaluation results
        """
        # Transcribe generated audio
        generated_text = self.transcribe_audio(generated_audio_path)
        
        # Calculate all metrics
        metrics = self.wer_calculator.calculate_all_metrics(reference_text, generated_text)
        
        # Get transcription with timestamps
        segments = self.transcribe_audio(generated_audio_path, with_timestamps=True)
        
        return {
            'reference_text': reference_text,
            'generated_text': generated_text,
            'segments': segments,
            'metrics': metrics,
            'model': self.model_name
        }
    
    def change_model(self, model_name: str):
        """
        Change the Whisper model
        
        Args:
            model_name: New model name
        """
        self.model_name = model_name
        self._load_model()
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get list of available models and their status
        
        Returns:
            Dictionary mapping model names to availability status
        """
        return self.model_manager.list_available_models()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return self.model_manager.get_model_info(self.model_name)
    
    @staticmethod
    def create_test_audio(duration: float = 5.0, 
                         noise_type: str = "white",
                         output_path: Optional[Union[str, Path]] = None) -> Union[np.ndarray, Path]:
        """
        Create test audio for evaluation
        
        Args:
            duration: Duration in seconds
            noise_type: Type of noise (white, pink, brown)
            output_path: Path to save audio file (optional)
            
        Returns:
            Audio data array or path to saved file
        """
        audio = generate_noise(duration, noise_type=noise_type)
        
        if output_path:
            save_audio_wav(audio, output_path)
            return Path(output_path)
        else:
            return audio
    
    @staticmethod
    def quick_wer(reference_text: str, 
                  generated_audio_path: Union[str, Path],
                  model: str = "base") -> float:
        """
        Quick WER calculation with default settings
        
        Args:
            reference_text: Reference text
            generated_audio_path: Path to generated audio
            model: Whisper model to use
            
        Returns:
            WER score
        """
        wm = WhisperMetrics(model=model)
        return wm.calculate_wer_with_reference(reference_text, generated_audio_path)