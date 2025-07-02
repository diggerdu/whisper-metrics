#!/usr/bin/env python3
"""
Build script for whisper.cpp bindings
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result

def build_whisper_cpp():
    """Build whisper.cpp and create Python bindings"""
    
    # Get paths
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    whisper_cpp_dir = build_dir / "whisper.cpp"
    package_dir = script_dir / "whisper_metrics"
    bin_dir = package_dir / "bin"
    
    # Create directories
    build_dir.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    
    # Clone whisper.cpp if not exists
    if not whisper_cpp_dir.exists():
        print("Cloning whisper.cpp...")
        run_command([
            "git", "clone", "--depth", "1", 
            "https://github.com/ggml-org/whisper.cpp.git",
            str(whisper_cpp_dir)
        ])
    
    # Build whisper.cpp
    print("Building whisper.cpp...")
    cmake_build_dir = build_dir / "cmake_build"
    cmake_build_dir.mkdir(exist_ok=True)
    
    # Configure
    run_command([
        "cmake",
        "-S", str(whisper_cpp_dir),
        "-B", str(cmake_build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
        "-DBUILD_SHARED_LIBS=ON",
    ])
    
    # Build
    run_command([
        "cmake", "--build", str(cmake_build_dir), 
        "--config", "Release",
        "--parallel"
    ])
    
    # Copy built library to package
    lib_files = list(cmake_build_dir.glob("libwhisper.*"))
    if not lib_files:
        lib_files = list(cmake_build_dir.glob("whisper.*"))
    
    if lib_files:
        for lib_file in lib_files:
            if lib_file.suffix in ['.so', '.dylib', '.dll']:
                dest = bin_dir / f"libwhisper{lib_file.suffix}"
                print(f"Copying {lib_file} -> {dest}")
                shutil.copy2(lib_file, dest)
                break
    else:
        print("Warning: No whisper library found in build directory")
    
    # Copy headers if needed
    header_src = whisper_cpp_dir / "whisper.h"
    header_dst = bin_dir / "whisper.h"
    if header_src.exists():
        shutil.copy2(header_src, header_dst)
    
    print("Build completed successfully!")

if __name__ == "__main__":
    build_whisper_cpp()