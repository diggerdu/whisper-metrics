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
    """Build whisper.cpp and GGML libraries from source during installation"""
    print("🔨 Building whisper.cpp and GGML from source...")
    print("   This may take a few minutes on first installation...")
    
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
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_SHARED=ON",  # Ensure GGML builds as shared library
            "-DWHISPER_BUILD_TESTS=OFF",
            "-DWHISPER_BUILD_EXAMPLES=OFF"
        ])
        
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