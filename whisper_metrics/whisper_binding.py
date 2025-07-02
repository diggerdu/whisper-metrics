"""
Python bindings for whisper.cpp using the proven approach from whisper-cpp-python
"""

from . import whisper_cpp
from typing import List, Optional, Union
import numpy as np
from pathlib import Path

class WhisperBinding:
    """Python binding for whisper.cpp using proven approach from whisper-cpp-python"""
    
    WHISPER_SR = 16000
    
    def __init__(self):
        self.context = None
        self.params = None
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load a whisper model from file"""
        try:
            if self.context is not None:
                whisper_cpp.whisper_free(self.context)
            
            self.context = whisper_cpp.whisper_init_from_file(str(model_path).encode('utf-8'))
            if self.context is None or self.context == 0:
                return False
            
            # Get default parameters
            self.params = whisper_cpp.whisper_full_default_params(0)  # GREEDY strategy
            
            # Configure parameters for stable transcription
            self.params.n_threads = 1
            self.params.print_special = False
            self.params.print_progress = False
            self.params.print_realtime = False
            self.params.print_timestamps = False
            self.params.translate = False
            self.params.no_context = False
            self.params.single_segment = False
            self.params.token_timestamps = False
            self.params.thold_pt = 0.01  # Probability threshold
            self.params.thold_ptsum = 0.01  # Sum of probabilities threshold
            self.params.max_len = 0  # No length limit
            self.params.split_on_word = False
            self.params.max_tokens = 0  # No token limit
            self.params.audio_ctx = 0  # Use full audio context
            self.params.vad = False  # Disable VAD
            self.params.vad_model_path = None
            self.params.language = b'en'  # Default to English
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """Transcribe audio data and return text"""
        if self.context is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            # Ensure audio data is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Set language if provided
            if language:
                self.params.language = language.encode('utf-8')
            
            # Run the inference using the same approach as whisper-cpp-python
            result = whisper_cpp.whisper_full(
                self.context, 
                self.params, 
                audio_data.ctypes.data_as(whisper_cpp.ctypes.POINTER(whisper_cpp.ctypes.c_float)), 
                len(audio_data)
            )
            
            if result != 0:
                return ""  # Return empty string on failure
            
            # Extract text from segments
            all_text = ''
            n_segments = whisper_cpp.whisper_full_n_segments(self.context)
            
            for i in range(n_segments):
                segment_text = whisper_cpp.whisper_full_get_segment_text(self.context, i)
                if segment_text:
                    text = segment_text.decode('utf-8')
                    all_text += text
            
            return all_text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray, language: Optional[str] = None) -> List[dict]:
        """Transcribe audio data and return text with timestamps"""
        if self.context is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            # Ensure audio data is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Set language if provided
            if language:
                self.params.language = language.encode('utf-8')
            
            # Run the inference
            result = whisper_cpp.whisper_full(
                self.context, 
                self.params, 
                audio_data.ctypes.data_as(whisper_cpp.ctypes.POINTER(whisper_cpp.ctypes.c_float)), 
                len(audio_data)
            )
            
            if result != 0:
                return []
            
            # Extract segments with timestamps
            segments = []
            n_segments = whisper_cpp.whisper_full_n_segments(self.context)
            
            for i in range(n_segments):
                t0 = whisper_cpp.whisper_full_get_segment_t0(self.context, i) / 100.0
                t1 = whisper_cpp.whisper_full_get_segment_t1(self.context, i) / 100.0
                segment_text = whisper_cpp.whisper_full_get_segment_text(self.context, i)
                
                if segment_text:
                    text = segment_text.decode('utf-8')
                    if text.strip():
                        segments.append({
                            'text': text.strip(),
                            'start': t0,
                            'end': t1
                        })
            
            return segments
            
        except Exception as e:
            print(f"Transcription with timestamps error: {e}")
            return []
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.context is not None:
            try:
                whisper_cpp.whisper_free(self.context)
            except:
                pass