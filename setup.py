"""
Setup configuration for Multimodal RAG System.

Provides package installation, dependency management, and entry points.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

# Development dependencies
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.1",
    "black>=23.7.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "isort>=5.12.0",
    "pre-commit>=3.3.3",
]

setup(
    name="multimodal-rag-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-grade Multimodal RAG system with safety guardrails",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-rag-system",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "rag-system=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml", "config/*.json"],
    },
    zip_safe=False,
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "multimodal",
        "nlp",
        "llm",
        "embeddings",
        "vector-search",
        "faiss",
        "openai",
        "safety-guardrails",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/multimodal-rag-system/issues",
        "Source": "https://github.com/yourusername/multimodal-rag-system",
        "Documentation": "https://github.com/yourusername/multimodal-rag-system/blob/main/README.md",
    },
)
