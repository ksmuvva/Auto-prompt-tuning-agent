from setuptools import setup, find_packages

setup(
    name="auto-prompt-tuning-agent",
    version="0.1.0",
    description="An AI agent that automatically tunes prompts for optimal performance",
    author="Auto Prompt Tuning Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "pytest-cov>=4.1.0",
            "pytest-timeout>=2.1.0",
            "hypothesis>=6.82.0",
            "freezegun>=1.2.0",
        ]
    },
)
