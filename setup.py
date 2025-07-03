#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.develop import develop

def detect_cuda():
    """Detect if CUDA is available on the system"""
    try:
        # Check if nvcc is available
        result = subprocess.run(['nvcc', '--version'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA detected - nvcc found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Check if CUDA libraries are available
    try:
        result = subprocess.run(['ldconfig', '-p'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'libcuda.so' in result.stdout:
            print("✅ CUDA detected - CUDA libraries found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    print("ℹ️  CUDA not detected - building CPU-only version")
    return False

def build_whisper_cpp():
    """Build whisper.cpp and GGML libraries from source during installation"""
    print("🔨 Building whisper.cpp and GGML from source...")
    print("   This may take a few minutes on first installation...")
    
    # Detect CUDA availability
    cuda_available = detect_cuda()
    
    # Get paths relative to setup.py
    script_dir = Path(__file__).parent.absolute()
    build_dir = script_dir / "build"
    whisper_cpp_dir = build_dir / "whisper.cpp"
    package_dir = script_dir / "whisper_metrics"
    bin_dir = package_dir / "bin"
    
    # Create directories
    build_dir.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    
    # Clone whisper.cpp if not exists
    if not whisper_cpp_dir.exists():
        print("📥 Cloning whisper.cpp repository...")
        try:
            subprocess.check_call([
                "git", "clone", "--depth", "1", 
                "https://github.com/ggml-org/whisper.cpp.git",
                str(whisper_cpp_dir)
            ])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone whisper.cpp: {e}")
            return False
    
    # Build whisper.cpp using cmake
    cmake_build_dir = build_dir / "cmake_build"
    cmake_build_dir.mkdir(exist_ok=True)
    
    try:
        # Configure with cmake
        cmake_args = [
            "cmake", 
            "-S", str(whisper_cpp_dir),
            "-B", str(cmake_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_SHARED=ON",  # Ensure GGML builds as shared library
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=OFF"
        ]
        
        # Add CUDA support if available
        if cuda_available:
            print("🚀 Configuring build with CUDA acceleration...")
            cmake_args.append("-DGGML_CUDA=1")
            # Try to detect GPU architecture automatically
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    compute_caps = result.stdout.strip().split('\n')
                    if compute_caps and compute_caps[0]:
                        # Convert compute capability to architecture number
                        compute_cap = compute_caps[0].replace('.', '')
                        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={compute_cap}")
                        print(f"🎯 Detected GPU compute capability: {compute_caps[0]} (arch {compute_cap})")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                print("ℹ️  Could not detect GPU architecture, using default CUDA architectures")
        else:
            print("🏗️  Configuring build for CPU-only...")
        
        subprocess.check_call(cmake_args)
        
        # Build both GGML and Whisper libraries
        print("🔧 Compiling GGML and Whisper...")
        subprocess.check_call([
            "cmake", "--build", str(cmake_build_dir), 
            "--config", "Release", 
            "--parallel", str(os.cpu_count() or 4)
        ])
        
        # Copy the compiled shared libraries to our bin directory
        print("📦 Copying compiled libraries...")
        
        # Find and copy all required shared libraries
        libraries_to_copy = []
        
        # Look for libwhisper.so in various locations
        whisper_locations = [
            cmake_build_dir / "src" / "libwhisper.so",
            cmake_build_dir / "libwhisper.so",
        ]
        
        lib_whisper = None
        for location in whisper_locations:
            if location.exists():
                lib_whisper = location
                break
        
        if lib_whisper:
            libraries_to_copy.append(("libwhisper.so", lib_whisper))
        else:
            print("❌ Could not find libwhisper.so")
            return False
        
        # Find all GGML-related libraries (they may be split into multiple files)
        print("🔍 Searching for all GGML libraries...")
        ggml_files = []
        
        # Search for all libggml*.so files
        for ggml_file in cmake_build_dir.rglob("libggml*.so"):
            ggml_files.append(ggml_file)
            print(f"   Found GGML library: {ggml_file}")
        
        if not ggml_files:
            print("❌ Could not find any GGML libraries")
            # List all .so files to help debug
            print("🔍 Available .so files in build directory:")
            for so_file in cmake_build_dir.rglob("*.so"):
                print(f"     {so_file}")
            return False
        
        # Add all GGML libraries to copy list
        for ggml_file in ggml_files:
            lib_name = ggml_file.name
            libraries_to_copy.append((lib_name, ggml_file))
        
        # Copy all libraries
        for lib_name, lib_path in libraries_to_copy:
            shutil.copy2(lib_path, bin_dir / lib_name)
            print(f"✅ Copied {lib_path} -> {bin_dir / lib_name}")
            
            # Set executable permissions
            os.chmod(bin_dir / lib_name, 0o755)
        
        # Copy header file
        header_file = whisper_cpp_dir / "include" / "whisper.h"
        if not header_file.exists():
            header_file = whisper_cpp_dir / "whisper.h"
        
        if header_file.exists():
            shutil.copy2(header_file, bin_dir / "whisper.h")
            print(f"✅ Copied {header_file} to {bin_dir}")
        
        print("🎉 whisper.cpp built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

# Global flag to track if whisper.cpp has been built
_whisper_built = False

class BuildWhisperCommand(build_py):
    """Custom build command to build whisper.cpp"""
    
    def run(self):
        global _whisper_built
        if not _whisper_built:
            build_whisper_cpp()
            _whisper_built = True
        super().run()

class InstallWhisperCommand(install):
    """Custom install command to ensure whisper.cpp is built"""
    
    def run(self):
        global _whisper_built
        if not _whisper_built:
            build_whisper_cpp()
            _whisper_built = True
        super().run()

class DevelopWhisperCommand(develop):
    """Custom develop command to ensure whisper.cpp is built"""
    
    def run(self):
        global _whisper_built
        if not _whisper_built:
            build_whisper_cpp()
            _whisper_built = True
        super().run()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper-metrics",
    version="0.1.0",
    author="Xingjian Du",
    author_email="xingjian.du97@gmail.com",
    description="Audio codec metrics calculation using whisper.cpp for transcription and WER evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diggerdu/whisper-metrics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "jiwer>=3.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    cmdclass={
        "build_py": BuildWhisperCommand,
        "install": InstallWhisperCommand,
        "develop": DevelopWhisperCommand,
    },
    zip_safe=False,
    include_package_data=True,
    package_data={
        "whisper_metrics": ["models/*", "bin/*"],
    },
)