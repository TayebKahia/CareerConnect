import os
import sys
import subprocess
import torch
from setuptools import setup, find_packages

# Get CUDA and PyTorch version information
CUDA_VERSION = torch.version.cuda
TORCH_VERSION = torch.__version__.split('.')
TORCH_MAJOR = TORCH_VERSION[0]
TORCH_MINOR = TORCH_VERSION[1]

print(f"PyTorch: {torch.__version__}, CUDA: {CUDA_VERSION}")

# Format version strings for PyG
if CUDA_VERSION:
    CUDA_STRING = f"+cu{CUDA_VERSION.replace('.', '')}"
else:
    CUDA_STRING = "+cpu"

TORCH_STRING = f"{TORCH_MAJOR}{TORCH_MINOR}"

# Define PyG components with version-specific wheels
PYG_WHEELS = [
    f"pyg-lib-0.3.0+pt{TORCH_STRING}{CUDA_STRING}",
    f"torch-scatter-2.1.2+pt{TORCH_STRING}{CUDA_STRING}",
    f"torch-sparse-0.6.18+pt{TORCH_STRING}{CUDA_STRING}",
    f"torch-cluster-1.6.3+pt{TORCH_STRING}{CUDA_STRING}",
    f"torch-spline-conv-1.2.2+pt{TORCH_STRING}{CUDA_STRING}"
]

# Install PyG components
print("Installing PyTorch Geometric components...")
for wheel in PYG_WHEELS:
    cmd = f"pip install --no-cache-dir {wheel} -f https://data.pyg.org/whl/{wheel}.html"
    print(f"Running: {cmd}")
    subprocess.call(cmd, shell=True)

# Install the main PyG package
subprocess.call("pip install torch-geometric", shell=True)

setup(
    name="job_recommendation_api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask==3.0.2',
        'flask-cors==4.0.0',
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'sentence-transformers>=2.2.2',
        'requests>=2.31.0',
        'nltk>=3.8.1',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'networkx>=3.0',
        'seaborn>=0.13.0',
        'tqdm>=4.66.0',
        'pillow>=10.0.0'
    ],
    python_requires='>=3.8',
)
