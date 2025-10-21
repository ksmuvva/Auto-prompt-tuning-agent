"""
Setup configuration for Prompt Tuning AI Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="prompt-tuning-agent",
    version="1.0.0",
    description="Automated Prompt Optimization AI Agent for Bank Transaction Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/Auto-prompt-tuning-agent",

    packages=find_packages(),

    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],

    extras_require={
        "openai": ["openai>=1.0.0", "tiktoken>=0.5.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tiktoken>=0.5.0",
            "jupyter>=1.0.0",
        ]
    },

    entry_points={
        "console_scripts": [
            "prompt-tuner=agent.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    python_requires=">=3.8",

    keywords="ai agent prompt-engineering llm optimization nlp machine-learning",
)
