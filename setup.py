#!/usr/bin/env python3
"""
Setup script for Learning Lens Blur Fields package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="learning-lens-blur-fields",
    version="1.0.0",
    author="Esther Y. H. Lin, Zhecheng Wang, Rebecca Lin, Daniel Miau, Florian Kainz, Jiawen Chen, Xuaner Cecilia Zhang, David B. Lindell, Kiriakos N. Kutulakos",
    author_email="[contact email]",
    description="Learning Lens Blur Fields - Neural representation of camera blur",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/learning-lens-blur-fields",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.pth", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "train-blur-field=scripts.train5d:main",
        ],
    },
) 