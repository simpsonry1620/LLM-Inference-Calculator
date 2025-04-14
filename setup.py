from setuptools import setup, find_packages

setup(
    name="advanced-calculator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "advanced-calculator=advanced_calculator.cli:main",
        ],
    },
    python_requires=">=3.8",
    author="Ryan Simpson",
    author_email="rsimpson@nvidia.com",
    description="Advanced calculator for LLM compute and memory requirements",
    keywords="llm, compute, memory, requirements",
    url="https://github.com/simpsonry1620/LLM-Inference-Calculator",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 