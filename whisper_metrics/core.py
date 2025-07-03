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
                 wer_config: Optional[Dict[str, Any]] = None,
                 use_gpu: Optional[bool] = None,
                 gpu_device: int = 0):
        """
        Initialize WhisperMetrics
        
        Args:
            model: Whisper model name (tiny, base, small, medium, large)
            auto_download: Automatically download model if not available
            wer_config: Configuration for WER calculator
            use_gpu: Whether to use GPU acceleration. If None, auto-detect based on availability
            gpu_device: GPU device ID to use (default: 0)
        """
        self.model_name = model
        self.auto_download = auto_download
        
        # Determine GPU usage
        if use_gpu is None:
            # Auto-detect GPU availability
            from . import is_cuda_available
            use_gpu = is_cuda_available()
        
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        
        # Initialize components
        self.model_manager = ModelManager()
        self.whisper_binding = WhisperBinding(use_gpu=use_gpu, gpu_device=gpu_device)
        
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
    
    def transcribe_numpy(self, 
                        audio_data: np.ndarray,
                        sample_rate: int = 16000,
                        with_timestamps: bool = False) -> Union[str, List[Dict]]:
        """
        Transcribe numpy audio array to text (zero file I/O overhead)
        
        Args:
            audio_data: Audio data as numpy array (float32, shape: [samples] or [samples, channels])
            sample_rate: Sample rate of the audio data (will resample to 16kHz if different)
            with_timestamps: Return timestamps with transcription
            
        Returns:
            Transcribed text or list of segments with timestamps
        """
        # Ensure audio data is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure audio data is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Handle multi-channel audio (convert to mono)
        if len(audio_data.shape) > 1:
            if audio_data.shape[1] > 1:  # Multi-channel
                audio_data = np.mean(audio_data, axis=1)
            elif audio_data.shape[0] > 1 and audio_data.shape[1] == 1:  # Single channel in 2D
                audio_data = audio_data.squeeze()
        
        # Resample if necessary
        if sample_rate != 16000:
            from scipy import signal
            # Calculate resampling ratio
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, target_length).astype(np.float32)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0 and max_val < 0.1:  # Boost very quiet audio
            audio_data = audio_data / max_val * 0.5
        
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
    
    def calculate_wer_with_reference_numpy(self, 
                                         reference_text: str,
                                         generated_audio_data: np.ndarray,
                                         sample_rate: int = 16000,
                                         return_all_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate WER between reference text and generated audio numpy array (zero file I/O overhead)
        
        Args:
            reference_text: Reference text transcription
            generated_audio_data: Generated audio data as numpy array
            sample_rate: Sample rate of the audio data
            return_all_metrics: Return all metrics (WER, MER, WIL, WIP, CER)
            
        Returns:
            WER score or dictionary of all metrics
        """
        # Transcribe generated audio
        generated_text = self.transcribe_numpy(generated_audio_data, sample_rate)
        
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
    
    def calculate_wer_numpy_to_numpy(self, 
                                   reference_audio_data: np.ndarray,
                                   generated_audio_data: np.ndarray,
                                   reference_sample_rate: int = 16000,
                                   generated_sample_rate: int = 16000,
                                   return_all_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate WER between transcriptions of reference and generated audio numpy arrays (zero file I/O overhead)
        
        Args:
            reference_audio_data: Reference audio data as numpy array
            generated_audio_data: Generated audio data as numpy array  
            reference_sample_rate: Sample rate of reference audio
            generated_sample_rate: Sample rate of generated audio
            return_all_metrics: Return all metrics (WER, MER, WIL, WIP, CER)
            
        Returns:
            WER score or dictionary of all metrics
        """
        # Transcribe both audio arrays
        reference_text = self.transcribe_numpy(reference_audio_data, reference_sample_rate)
        generated_text = self.transcribe_numpy(generated_audio_data, generated_sample_rate)
        
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
    
    def batch_calculate_wer_with_reference_numpy(self, 
                                               reference_texts: List[str],
                                               generated_audio_arrays: List[np.ndarray],
                                               sample_rates: Union[int, List[int]] = 16000,
                                               return_all_metrics: bool = False) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate WER for multiple audio numpy arrays with reference texts (zero file I/O overhead)
        
        Args:
            reference_texts: List of reference text transcriptions
            generated_audio_arrays: List of generated audio numpy arrays
            sample_rates: Sample rate(s) - single int for all, or list matching audio arrays
            return_all_metrics: Return all metrics for each file
            
        Returns:
            List of WER scores or list of metric dictionaries
        """
        if len(reference_texts) != len(generated_audio_arrays):
            raise ValueError("Number of reference texts must match number of generated audio arrays")
        
        # Handle sample rates
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates] * len(generated_audio_arrays)
        elif len(sample_rates) != len(generated_audio_arrays):
            raise ValueError("Number of sample rates must match number of audio arrays or be a single int")
        
        results = []
        for ref_text, audio_data, sr in zip(reference_texts, generated_audio_arrays, sample_rates):
            result = self.calculate_wer_with_reference_numpy(ref_text, audio_data, sr, return_all_metrics)
            results.append(result)
        
        return results
    
    def batch_calculate_wer_numpy_to_numpy(self, 
                                         reference_audio_arrays: List[np.ndarray],
                                         generated_audio_arrays: List[np.ndarray],
                                         reference_sample_rates: Union[int, List[int]] = 16000,
                                         generated_sample_rates: Union[int, List[int]] = 16000,
                                         return_all_metrics: bool = False) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate WER for multiple pairs of audio numpy arrays (zero file I/O overhead)
        
        Args:
            reference_audio_arrays: List of reference audio numpy arrays
            generated_audio_arrays: List of generated audio numpy arrays
            reference_sample_rates: Sample rate(s) for reference audio
            generated_sample_rates: Sample rate(s) for generated audio
            return_all_metrics: Return all metrics for each pair
            
        Returns:
            List of WER scores or list of metric dictionaries
        """
        if len(reference_audio_arrays) != len(generated_audio_arrays):
            raise ValueError("Number of reference and generated audio arrays must match")
        
        # Handle sample rates
        if isinstance(reference_sample_rates, int):
            reference_sample_rates = [reference_sample_rates] * len(reference_audio_arrays)
        if isinstance(generated_sample_rates, int):
            generated_sample_rates = [generated_sample_rates] * len(generated_audio_arrays)
        
        if len(reference_sample_rates) != len(reference_audio_arrays):
            raise ValueError("Number of reference sample rates must match number of reference arrays or be a single int")
        if len(generated_sample_rates) != len(generated_audio_arrays):
            raise ValueError("Number of generated sample rates must match number of generated arrays or be a single int")
        
        results = []
        for ref_data, gen_data, ref_sr, gen_sr in zip(reference_audio_arrays, generated_audio_arrays, 
                                                      reference_sample_rates, generated_sample_rates):
            result = self.calculate_wer_numpy_to_numpy(ref_data, gen_data, ref_sr, gen_sr, return_all_metrics)
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
    
    def evaluate_with_detailed_output_numpy(self, 
                                          reference_text: str,
                                          generated_audio_data: np.ndarray,
                                          sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Evaluate numpy audio with detailed output including transcriptions and all metrics (zero file I/O overhead)
        
        Args:
            reference_text: Reference text transcription
            generated_audio_data: Generated audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Dictionary with detailed evaluation results
        """
        # Transcribe generated audio
        generated_text = self.transcribe_numpy(generated_audio_data, sample_rate)
        
        # Calculate all metrics
        metrics = self.wer_calculator.calculate_all_metrics(reference_text, generated_text)
        
        # Get transcription with timestamps
        segments = self.transcribe_numpy(generated_audio_data, sample_rate, with_timestamps=True)
        
        return {
            'reference_text': reference_text,
            'generated_text': generated_text,
            'segments': segments,
            'metrics': metrics,
            'model': self.model_name,
            'audio_shape': generated_audio_data.shape,
            'sample_rate': sample_rate
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
    
    def set_gpu_usage(self, use_gpu: bool, gpu_device: int = 0):
        """
        Toggle GPU usage. This will reload the model with new GPU settings.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            gpu_device: GPU device ID to use (default: 0)
        """
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        
        # Update whisper binding GPU settings
        self.whisper_binding.set_gpu_usage(use_gpu, gpu_device)
        
        # Reload the model with new GPU settings
        self._load_model()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU configuration and status
        
        Returns:
            Dictionary with GPU status information
        """
        from . import is_cuda_available
        
        binding_status = self.whisper_binding.get_gpu_status()
        
        return {
            'requested_gpu_usage': self.use_gpu,
            'gpu_device': self.gpu_device,
            'cuda_available': is_cuda_available(),
            'binding_status': binding_status,
            'model_loaded': binding_status['context_loaded']
        }
    
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
                  model: str = "base",
                  use_gpu: Optional[bool] = None) -> float:
        """
        Quick WER calculation with default settings
        
        Args:
            reference_text: Reference text
            generated_audio_path: Path to generated audio
            model: Whisper model to use
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
            
        Returns:
            WER score
        """
        wm = WhisperMetrics(model=model, use_gpu=use_gpu)
        return wm.calculate_wer_with_reference(reference_text, generated_audio_path)
    
    @staticmethod
    def quick_wer_numpy(reference_text: str, 
                       generated_audio_data: np.ndarray,
                       sample_rate: int = 16000,
                       model: str = "base",
                       use_gpu: Optional[bool] = None) -> float:
        """
        Quick WER calculation with numpy array (zero file I/O overhead)
        
        Args:
            reference_text: Reference text
            generated_audio_data: Generated audio data as numpy array
            sample_rate: Sample rate of the audio data
            model: Whisper model to use
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
            
        Returns:
            WER score
        """
        wm = WhisperMetrics(model=model, use_gpu=use_gpu)
        return wm.calculate_wer_with_reference_numpy(reference_text, generated_audio_data, sample_rate)