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

def build_whisper_cpp():
    """Build whisper.cpp and install binaries"""
    print("🔨 Building whisper.cpp...")
    
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
        print("🏗️  Configuring build...")
        subprocess.check_call([
            "cmake", 
            "-S", str(whisper_cpp_dir),
            "-B", str(cmake_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON"
        ])
        
        # Build
        print("🔧 Compiling...")
        subprocess.check_call([
            "cmake", "--build", str(cmake_build_dir), 
            "--config", "Release", "-j", str(os.cpu_count() or 4)
        ])
        
        # Copy the shared library to our bin directory
        lib_file = cmake_build_dir / "src" / "libwhisper.so"
        if not lib_file.exists():
            lib_file = cmake_build_dir / "libwhisper.so"
        
        if lib_file.exists():
            shutil.copy2(lib_file, bin_dir / "libwhisper.so")
            print(f"✅ Copied {lib_file} to {bin_dir}")
        else:
            print("❌ Could not find libwhisper.so")
            return False
        
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

class BuildWhisperCommand(build_py):
    """Custom build command to build whisper.cpp"""
    
    def run(self):
        build_whisper_cpp()
        super().run()

class InstallWhisperCommand(install):
    """Custom install command to ensure whisper.cpp is built"""
    
    def run(self):
        build_whisper_cpp()
        super().run()

class DevelopWhisperCommand(develop):
    """Custom develop command to ensure whisper.cpp is built"""
    
    def run(self):
        build_whisper_cpp()
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