#!/usr/bin/env python3
"""
Setup script for Linux AI Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="linux-ai-agent",
    version="1.0.0",
    author="Linux AI Agent Team",
    description="A modular Python-based AI agent for Linux file operations, system monitoring, and malware detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/linux-ai-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "Topic :: Security",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "ai": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "spacy>=3.4.0",
            "nltk>=3.7",
            "sentence-transformers>=2.2.0",
            "tensorflow>=2.9.0",
            "gymnasium>=0.26.0",
            "stable-baselines3>=1.6.0",
        ],
        "enhanced": [
            "chromadb>=0.3.0",
            "faiss-cpu>=1.7.0",
            "google-generativeai>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "linux-ai-agent=main:main",
            "lai=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)