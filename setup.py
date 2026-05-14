"""
NysAdam: Nyström-Augmented Adaptive Moment Estimation
=====================================================
Author: Hardik Tankaria
Date: May 2026
Description: A regularized Nyström-preconditioned optimizer for non-convex
             optimization in ML, DL, and LLMs.
"""

from setuptools import setup, find_packages

setup(
    name="nysadam",
    version="0.1.0",
    author="Hardik Tankaria",
    description="NysAdam: Nyström-Augmented Adaptive Moment Estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "wandb>=0.15.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "llm": ["transformers>=4.30.0", "datasets>=2.13.0", "accelerate>=0.20.0"],
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
)
