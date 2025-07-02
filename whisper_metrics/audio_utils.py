"""
Audio utilities for loading and preprocessing audio files
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

def load_audio_numpy(file_path: Union[str, Path], 
                    target_sr: int = 16000,
                    mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file using numpy and basic audio processing
    
    This is a basic implementation that works with WAV files.
    For production use, consider using librosa or soundfile.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(file_path)
        
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        
        # Convert to mono if needed
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed (basic nearest neighbor resampling)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        
        return audio, sr
        
    except ImportError:
        raise ImportError("scipy is required for basic audio loading. Install with: pip install scipy")

def resample_audio(audio: np.ndarray, 
                  original_sr: int, 
                  target_sr: int) -> np.ndarray:
    """
    Basic audio resampling using linear interpolation
    
    For better quality, consider using librosa.resample()
    
    Args:
        audio: Audio data
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if original_sr == target_sr:
        return audio
    
    # Calculate the ratio
    ratio = target_sr / original_sr
    
    # Calculate new length
    new_length = int(len(audio) * ratio)
    
    # Create new time indices
    original_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    # Interpolate
    resampled = np.interp(new_indices, original_indices, audio)
    
    return resampled

def try_load_with_librosa(file_path: Union[str, Path], 
                         target_sr: int = 16000,
                         mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Try to load audio with librosa (preferred method)
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except ImportError:
        raise ImportError("librosa not available. Install with: pip install librosa")

def try_load_with_soundfile(file_path: Union[str, Path], 
                           target_sr: int = 16000,
                           mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Try to load audio with soundfile
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path)
        
        # Convert to float32
        audio = audio.astype(np.float32)
        
        # Convert to mono if needed
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr
        
        return audio, sr
        
    except ImportError:
        raise ImportError("soundfile not available. Install with: pip install soundfile")

def load_audio(file_path: Union[str, Path], 
               target_sr: int = 16000,
               mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file with fallback options
    
    Tries multiple audio loading libraries in order of preference:
    1. librosa (best quality)
    2. soundfile (good quality)
    3. scipy.io.wavfile (basic, WAV only)
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000 Hz for Whisper)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        ImportError: If no audio loading library is available
        FileNotFoundError: If audio file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Try librosa first (best quality)
    try:
        return try_load_with_librosa(file_path, target_sr, mono)
    except ImportError:
        pass
    
    # Try soundfile
    try:
        return try_load_with_soundfile(file_path, target_sr, mono)
    except ImportError:
        pass
    
    # Fall back to scipy (basic WAV support)
    try:
        return load_audio_numpy(file_path, target_sr, mono)
    except ImportError:
        pass
    
    raise ImportError(
        "No audio loading library available. Please install one of:\n"
        "- librosa: pip install librosa\n"
        "- soundfile: pip install soundfile\n"
        "- scipy: pip install scipy"
    )

def generate_noise(duration: float, 
                  sample_rate: int = 16000,
                  noise_type: str = "white") -> np.ndarray:
    """
    Generate noise audio for testing
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        noise_type: Type of noise ("white", "pink", "brown")
        
    Returns:
        Noise audio data
    """
    n_samples = int(duration * sample_rate)
    
    if noise_type == "white":
        # White noise (equal power at all frequencies)
        return np.random.normal(0, 0.1, n_samples).astype(np.float32)
    
    elif noise_type == "pink":
        # Pink noise (1/f power spectrum)
        # Generate white noise and apply pink filter
        white = np.random.normal(0, 1, n_samples)
        # Simple pink noise approximation
        pink = np.zeros_like(white)
        for i in range(1, len(white)):
            pink[i] = 0.99886 * pink[i-1] + white[i] * 0.0555179
        return (pink * 0.1).astype(np.float32)
    
    elif noise_type == "brown":
        # Brown noise (1/f^2 power spectrum)
        white = np.random.normal(0, 1, n_samples)
        brown = np.zeros_like(white)
        for i in range(1, len(white)):
            brown[i] = brown[i-1] + white[i] * 0.1
        # Normalize
        brown = brown / np.std(brown) * 0.1
        return brown.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def save_audio_wav(audio: np.ndarray, 
                   file_path: Union[str, Path],
                   sample_rate: int = 16000) -> None:
    """
    Save audio as WAV file
    
    Args:
        audio: Audio data
        file_path: Output file path
        sample_rate: Sample rate
    """
    try:
        import scipy.io.wavfile as wavfile
        
        # Convert to int16 for WAV format
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        wavfile.write(file_path, sample_rate, audio_int16)
        
    except ImportError:
        raise ImportError("scipy is required for WAV file saving. Install with: pip install scipy")