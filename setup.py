from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nanoformula",
    version="2.0.0",
    author="Hardik Sood, Dr. Ruchi Chawla",
    author_email="hardik.sood.phe24@itbhu.ac.in",
    description="Machine Learning-Driven Multi-Polymer Nanoparticle Formulation Optimizer & Virtual Screening Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hardiksood21/nanoformula-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "scipy>=1.11.0",
        "shap>=0.44.0",
        "rdkit>=2023.9.0",
        "reportlab>=4.0.0",
        "plotly>=5.18.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "nanoformula=app:main",
        ],
    },
)
