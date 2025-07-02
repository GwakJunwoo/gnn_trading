"""
Setup configuration for GNN Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

# Read development requirements
dev_requirements_path = Path(__file__).parent / "requirements-dev.txt"
dev_requirements = []
if dev_requirements_path.exists():
    with open(dev_requirements_path, "r", encoding="utf-8") as f:
        dev_requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="gnn-trading",
    version="2.0.0",
    author="GNN Trading Team",
    author_email="lunarsabbth@gmail.com",
    description="Advanced Graph Neural Network Trading System for Korean Financial Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GwakJunwoo/gnn-trading",
    project_urls={
        "Bug Tracker": "https://github.com/GwakJunwoo/gnn-trading/issues",
        "Documentation": "https://gnn-trading.readthedocs.io/",
        "Source Code": "https://github.com/your-org/gnn-trading",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "gnn_trading": [
            "configs/*.yaml",
            "data/templates/*.csv",
            "docs/*.md",
        ],
    },
    
    # Dependencies
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": [
            "torch[gpu]",
            "cupy-cuda11x",
            "cudf-cu11",
        ],
        "korea": [
            "beautifulsoup4>=4.12.0",
            "selenium>=4.10.0",
            "requests-html>=0.10.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
            "elasticsearch>=8.8.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "gnn-trading=gnn_trading.cli:main",
            "gnn-trading-api=gnn_trading.api.main:main",
            "gnn-trading-train=gnn_trading.models.trainer:main",
            "gnn-trading-backtest=gnn_trading.backtest.engine:main",
            "gnn-trading-quality=gnn_trading.data_pipeline.quality:main",
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "machine learning",
        "graph neural networks",
        "algorithmic trading", 
        "financial modeling",
        "korean markets",
        "time series",
        "ensemble learning",
        "real-time processing",
    ],
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    zip_safe=False,
)
