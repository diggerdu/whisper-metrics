# Whisper Metrics

A Python package for calculating audio codec metrics using whisper.cpp for transcription and Word Error Rate (WER) evaluation.

## Features

- **Battery-included**: Automatically compiles whisper.cpp binaries and downloads model checkpoints
- **Dual-mode operation**: Calculate WER with or without reference transcriptions
- **High-performance**: Uses whisper.cpp C++ backend for fast transcription
- **Linux x86-64 optimized**: Specifically designed for Linux x86-64 platforms

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/diggerdu/whisper-metrics.git
```

### Requirements

- Python 3.8+
- Linux x86-64
- CMake (for building whisper.cpp)
- GCC/G++ compiler

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake build-essential

# CentOS/RHEL
sudo yum install cmake gcc-c++ make
```

## Usage

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