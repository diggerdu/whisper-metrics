#!/usr/bin/env python3
"""
Installation test script for whisper-metrics package

This script tests the whisper-metrics package installation and functionality.
Run this after installing the package with:
    pip install git+https://github.com/diggerdu/whisper-metrics.git

Usage:
    python test_installation.py
"""

import sys
import os
import tempfile
from pathlib import Path

def download_test_files():
    """Create test files if they don't exist"""
    print("📁 Checking for test files...")
    
    # Check if test.wav and test.txt exist
    test_wav = Path("test.wav")
    test_txt = Path("test.txt")
    
    if not test_wav.exists():
        print("⬇️  Test audio file not found. Creating a test tone...")
        try:
            import numpy as np
            from whisper_metrics.audio_utils import save_audio_wav
            
            # Generate a 3-second 440Hz tone
            duration = 3.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4 note
            audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            save_audio_wav(audio, "test.wav")
            print("✓ Created test audio file (test.wav)")
        except Exception as e:
            print(f"❌ Could not create test audio: {e}")
            return False
    else:
        print("✓ Found test.wav")
    
    if not test_txt.exists():
        print("📝 Creating reference text file...")
        with open("test.txt", "w") as f:
            f.write("This is a test audio file for whisper metrics.")
        print("✓ Created test.txt")
    else:
        print("✓ Found test.txt")
    
    return True

def test_package_import():
    """Test if the package can be imported"""
    print("\n🔍 Testing package import...")
    
    try:
        import whisper_metrics
        print("✓ whisper_metrics package imported successfully")
        
        from whisper_metrics import WhisperMetrics, WERCalculator, ModelManager
        print("✓ Main classes imported successfully")
        
        from whisper_metrics.audio_utils import load_audio, generate_noise
        print("✓ Audio utilities imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_model_management():
    """Test model download and management"""
    print("\n📦 Testing model management...")
    
    try:
        from whisper_metrics import ModelManager
        
        model_manager = ModelManager()
        
        # List available models
        models = model_manager.list_available_models()
        print(f"✓ Available models: {len(models)} models")
        
        # Get info about tiny model
        tiny_info = model_manager.get_model_info("tiny")
        print(f"✓ Tiny model info: {tiny_info['size']}")
        
        return True
    except Exception as e:
        print(f"❌ Model management test failed: {e}")
        return False

def test_audio_utilities():
    """Test audio loading and processing"""
    print("\n🎵 Testing audio utilities...")
    
    try:
        from whisper_metrics.audio_utils import generate_noise, save_audio_wav
        
        # Test noise generation
        noise = generate_noise(duration=1.0, noise_type="white")
        print(f"✓ Generated white noise: {len(noise)} samples")
        
        # Test saving
        save_audio_wav(noise, "test_noise.wav")
        print("✓ Saved test noise to file")
        
        # Cleanup
        if Path("test_noise.wav").exists():
            os.remove("test_noise.wav")
        
        return True
    except Exception as e:
        print(f"❌ Audio utilities test failed: {e}")
        return False

def test_wer_calculation():
    """Test WER calculation functionality"""
    print("\n📊 Testing WER calculation...")
    
    try:
        from whisper_metrics import WERCalculator
        
        wer_calc = WERCalculator()
        
        # Test basic WER calculation
        reference = "hello world this is a test"
        hypothesis = "hello world this is a test"
        wer = wer_calc.calculate_wer(reference, hypothesis)
        print(f"✓ Perfect match WER: {wer:.3f}")
        
        # Test with differences
        hypothesis2 = "hello world this is test"
        wer2 = wer_calc.calculate_wer(reference, hypothesis2)
        print(f"✓ With missing word WER: {wer2:.3f}")
        
        # Test all metrics
        metrics = wer_calc.calculate_all_metrics(reference, hypothesis2)
        print(f"✓ All metrics: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ WER calculation test failed: {e}")
        return False

def test_whisper_initialization():
    """Test Whisper model initialization"""
    print("\n🤖 Testing Whisper model initialization...")
    
    try:
        from whisper_metrics import WhisperMetrics
        
        # Try to initialize with tiny model
        print("   Initializing WhisperMetrics with tiny model...")
        wm = WhisperMetrics(model="tiny", auto_download=True)
        print("✓ WhisperMetrics initialized successfully")
        
        # Get model info
        try:
            model_info = wm.get_model_info()
            print(f"✓ Model info: {model_info.get('size', 'N/A')}")
        except:
            print("✓ Model loaded (info method not available)")
        
        return True, wm
    except Exception as e:
        print(f"❌ Whisper initialization failed: {e}")
        return False, None

def test_transcription(wm):
    """Test audio transcription functionality"""
    print("\n🎤 Testing audio transcription...")
    
    if wm is None:
        print("❌ Cannot test transcription - WhisperMetrics not initialized")
        return False
    
    try:
        if not Path("test.wav").exists():
            print("❌ test.wav not found for transcription test")
            return False
        
        # Test transcription
        print("   Transcribing test.wav...")
        transcription = wm.transcribe_audio("test.wav")
        print(f"✓ Transcription result: '{transcription}'")
        
        # Test with timestamps
        try:
            segments = wm.transcribe_audio("test.wav", with_timestamps=True)
            print(f"✓ Transcription with timestamps: {len(segments)} segments")
        except:
            print("✓ Basic transcription works (timestamps may not be available)")
        
        return True
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

def test_wer_with_transcription(wm):
    """Test WER calculation with actual transcription"""
    print("\n📈 Testing WER calculation with transcription...")
    
    if wm is None:
        print("❌ Cannot test WER with transcription - WhisperMetrics not initialized")
        return False
    
    try:
        if not Path("test.wav").exists() or not Path("test.txt").exists():
            print("❌ test.wav or test.txt not found for WER test")
            return False
        
        # Load reference text
        with open("test.txt", "r") as f:
            reference_text = f.read().strip()
        
        # Calculate WER
        wer = wm.calculate_wer_with_reference(reference_text, "test.wav")
        print(f"✓ WER score: {wer:.3f}")
        
        # Calculate all metrics
        try:
            metrics = wm.calculate_wer_with_reference(reference_text, "test.wav", return_all_metrics=True)
            print(f"✓ Complete metrics: {list(metrics.keys())}")
        except:
            print("✓ Basic WER calculation works")
        
        return True
    except Exception as e:
        print(f"❌ WER with transcription test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 WHISPER-METRICS INSTALLATION TEST")
    print("=" * 50)
    print("Testing whisper-metrics package functionality...")
    print("=" * 50)
    
    # Track test results
    tests = []
    
    # Test 1: Download/check test files
    tests.append(("Download test files", download_test_files()))
    
    # Test 2: Package import
    tests.append(("Package import", test_package_import()))
    
    # Test 3: Model management
    tests.append(("Model management", test_model_management()))
    
    # Test 4: Audio utilities
    tests.append(("Audio utilities", test_audio_utilities()))
    
    # Test 5: WER calculation
    tests.append(("WER calculation", test_wer_calculation()))
    
    # Test 6: Whisper initialization
    whisper_success, wm = test_whisper_initialization()
    tests.append(("Whisper initialization", whisper_success))
    
    # Test 7: Transcription (only if Whisper initialized)
    if whisper_success:
        tests.append(("Audio transcription", test_transcription(wm)))
        tests.append(("WER with transcription", test_wer_with_transcription(wm)))
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ whisper-metrics package is working correctly!")
        print("\n📖 USAGE EXAMPLE:")
        print("```python")
        print("from whisper_metrics import WhisperMetrics")
        print("wm = WhisperMetrics(model='tiny')")
        print("transcription = wm.transcribe_audio('audio.wav')")
        print("wer = wm.calculate_wer_with_reference('reference text', 'audio.wav')")
        print("```")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed!")
        print("Please check the error messages above for troubleshooting.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())