#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

class BuildWhisperCommand(build_py):
    """Custom build command to build whisper.cpp"""
    
    def run(self):
        # Run the whisper.cpp build script
        build_script = Path(__file__).parent / "build_whisper.py"
        if build_script.exists():
            try:
                subprocess.check_call([sys.executable, str(build_script)])
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to build whisper.cpp: {e}")
                print("The package will be installed without whisper.cpp bindings.")
        
        # Run the normal build
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
    cmdclass={"build_py": BuildWhisperCommand},
    zip_safe=False,
    include_package_data=True,
    package_data={
        "whisper_metrics": ["models/*", "bin/*"],
    },
)