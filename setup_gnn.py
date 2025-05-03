from setuptools import setup, find_packages

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
        'requests>=2.31.0'
    ],
    python_requires='>=3.8',
)