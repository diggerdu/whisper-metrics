# Whisper Metrics

A Python package for calculating audio codec metrics using whisper.cpp for transcription and Word Error Rate (WER) evaluation.

## Features

- **Battery-included**: Automatically compiles whisper.cpp binaries and downloads model checkpoints
- **Dual-mode operation**: Calculate WER with or without reference transcriptions
- **High-performance**: Uses whisper.cpp C++ backend for fast transcription
- **CUDA acceleration**: Automatic GPU acceleration when CUDA is available
- **Linux x86-64 optimized**: Specifically designed for Linux x86-64 platforms

## Installation

### Quick Install

**CPU-only version (default):**
```bash
pip install git+https://github.com/diggerdu/whisper-metrics.git
```

**With CUDA acceleration:**
```bash
# Method 1: Using environment variable (recommended)
WHISPER_METRICS_CUDA=1 pip install git+https://github.com/diggerdu/whisper-metrics.git

# Method 2: Using pip extras (experimental)
pip install git+https://github.com/diggerdu/whisper-metrics.git#egg=whisper-metrics[cuda]
```

**Note**: The first installation will take a few minutes as it builds whisper.cpp from source.

### Requirements

- Python 3.8+
- Linux x86-64
- CMake (for building whisper.cpp)
- GCC/G++ compiler
- Git

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install cmake build-essential git
```

**CentOS/RHEL:**
```bash
sudo yum install cmake gcc-c++ make git
```

**For older systems, you may need:**
```bash
# Ubuntu/Debian
sudo apt install python3-dev

# CentOS/RHEL  
sudo yum install python3-devel
```

### CUDA Support (Optional)

For GPU acceleration, you need to:

1. **Install CUDA toolkit** before installing whisper-metrics:

**Using Conda (Recommended):**
```bash
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
```

**Using NVIDIA Installation:**
- Download CUDA from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
- Follow the installation guide for your Linux distribution

2. **Install with CUDA support:**
```bash
# Method 1: Using environment variable (recommended)
WHISPER_METRICS_CUDA=1 pip install git+https://github.com/diggerdu/whisper-metrics.git

# Method 2: Using pip extras (experimental)
pip install git+https://github.com/diggerdu/whisper-metrics.git#egg=whisper-metrics[cuda]
```

**Requirements for CUDA:**
- NVIDIA GPU with compute capability 5.0 or higher
- NVIDIA driver version 450.80.02 or higher
- CUDA toolkit 11.0 or higher

**Note**: The `[cuda]` extra forces CUDA compilation and requires CUDA to be installed. The default installation is CPU-only.

### Installation Process

The installation automatically:
1. 📥 Downloads whisper.cpp source code
2. 🔨 Compiles the C++ library with optimizations
3. 📦 Installs Python bindings and dependencies
4. ✅ Sets up the complete package

### Troubleshooting

**If installation fails:**

1. **Missing dependencies**: Install system dependencies above
2. **Build errors**: Ensure you have sufficient disk space (>1GB)
3. **Permission errors**: Use `pip install --user` for user-only installation
4. **Network issues**: Check internet connection for downloading whisper.cpp

**Force reinstall:**
```bash
# CPU-only version
pip uninstall whisper-metrics
pip install --force-reinstall git+https://github.com/diggerdu/whisper-metrics.git

# CUDA version
pip uninstall whisper-metrics
WHISPER_METRICS_CUDA=1 pip install --force-reinstall git+https://github.com/diggerdu/whisper-metrics.git
```

## Usage

### Checking CUDA Availability

```python
import whisper_metrics

# Check if CUDA acceleration is available
if whisper_metrics.is_cuda_available():
    print("🚀 CUDA acceleration is available!")
else:
    print("💻 Running in CPU-only mode")
```

### Mode 1: With Reference Transcription

Calculate WER between reference text and transcribed audio:

```python
from whisper_metrics import WhisperMetrics

# Initialize with model (automatically downloads if needed)
wm = WhisperMetrics(model="base")

# Calculate WER with reference text
reference_text = "Hello world, this is a test"
generated_audio_path = "generated_audio.wav"

wer_score = wm.calculate_wer_with_reference(
    reference_text=reference_text,
    generated_audio_path=generated_audio_path
)

print(f"Word Error Rate: {wer_score:.3f}")
```

### Mode 2: Without Reference Transcription

Calculate WER by transcribing both reference and generated audio:

```python
from whisper_metrics import WhisperMetrics

wm = WhisperMetrics(model="base")

# Calculate WER by transcribing both audios
reference_audio_path = "reference_audio.wav"
generated_audio_path = "generated_audio.wav"

wer_score = wm.calculate_wer_audio_to_audio(
    reference_audio_path=reference_audio_path,
    generated_audio_path=generated_audio_path
)

print(f"Word Error Rate: {wer_score:.3f}")
```

### Direct Transcription

```python
# Just transcribe audio to text
transcription = wm.transcribe_audio("audio.wav")
print(f"Transcription: {transcription}")

# With timestamps
segments = wm.transcribe_audio("audio.wav", with_timestamps=True)
for segment in segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

## Testing Installation

After installation, test the package functionality:

```bash
# Download the test script
curl -O https://raw.githubusercontent.com/diggerdu/whisper-metrics/main/test_installation.py

# Run the test with your own audio files
python test_installation.py
```

The test script will verify:
- Package installation
- Model downloading
- Audio transcription
- WER calculation
- All core functionality

## Available Models

| Model | Size | Speed | Quality | Description |
|-------|------|-------|---------|-------------|
| tiny  | 39 MB | ~32x | Good | Fastest, good for testing |
| base  | 74 MB | ~16x | Better | Recommended for most uses |
| small | 244 MB | ~6x | Better | More accurate transcription |
| medium| 769 MB | ~2x | Best | High accuracy, slower |
| large | 1550 MB | 1x | Best | Highest accuracy, slowest |

## API Reference

### WhisperMetrics Class

```python
WhisperMetrics(model="base", auto_download=True, wer_config=None)
```

**Parameters:**
- `model`: Model size ("tiny", "base", "small", "medium", "large")
- `auto_download`: Automatically download models if not available
- `wer_config`: Configuration for WER calculator

**Methods:**

- `transcribe_audio(audio_path, with_timestamps=False)` - Transcribe audio file
- `calculate_wer_with_reference(reference_text, audio_path, return_all_metrics=False)` - Calculate WER with reference text
- `calculate_wer_audio_to_audio(ref_audio, gen_audio, return_all_metrics=False)` - Calculate WER between two audio files
- `batch_calculate_wer_with_reference(ref_texts, audio_paths, return_all_metrics=False)` - Batch processing

### Metrics

When `return_all_metrics=True`, you get:
- **WER**: Word Error Rate
- **MER**: Match Error Rate
- **WIL**: Word Information Lost
- **WIP**: Word Information Preserved  
- **CER**: Character Error Rate

## License

MIT License - see LICENSE file for details.

## Author

Xingjian Du (xingjian.du97@gmail.com)