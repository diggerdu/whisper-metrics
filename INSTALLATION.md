# Installation Guide for whisper-metrics

## Prerequisites

- Linux x86-64 system
- Python 3.8 or higher
- CMake (for building whisper.cpp)
- Git
- GCC/G++ compiler

## Installation Methods

### Method 1: Development Installation

1. Clone or download the package:
```bash
cd whisper-metrics
```

2. Install dependencies:
```bash
pip install numpy requests tqdm jiwer scipy
```

3. Build whisper.cpp bindings:
```bash
python build_whisper.py
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Method 2: Production Installation

```bash
pip install .
```

This will automatically:
- Install all required dependencies
- Build whisper.cpp binaries
- Download model checkpoints when needed

## Dependencies

### Required
- `numpy>=1.20.0` - Numerical computing
- `requests>=2.25.0` - HTTP requests for model downloads
- `tqdm>=4.60.0` - Progress bars
- `jiwer>=3.0.0` - Word Error Rate calculation
- `scipy>=1.7.0` - Audio file handling

### Optional (for better audio support)
- `librosa` - High-quality audio processing
- `soundfile` - Audio file I/O

## Verification

Run the test suite to verify installation:

```bash
python test_whisper_metrics.py
```

Run the demonstration:

```bash
python demo.py
```

## Troubleshooting

### CMake not found
```bash
# Ubuntu/Debian
sudo apt-get install cmake

# CentOS/RHEL
sudo yum install cmake
```

### GCC not found
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

### Audio loading issues
Install additional audio libraries:
```bash
pip install librosa soundfile
```

## Model Downloads

Models are automatically downloaded to `whisper_metrics/models/` directory on first use.

Available models:
- `tiny` (39 MB) - Fastest, least accurate
- `base` (142 MB) - Good balance
- `small` (466 MB) - Better accuracy
- `medium` (1.5 GB) - High accuracy
- `large` (2.9 GB) - Best accuracy

## License

MIT License - see LICENSE file for details.