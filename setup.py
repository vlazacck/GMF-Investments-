from setuptools import setup, find_packages

setup(
    name="dashboard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas==2.1.0",
        "numpy==1.24.3",
        "matplotlib==3.7.1",
        "statsmodels==0.14.0",
        "yfinance==0.2.28",
        "seaborn==0.12.2",
        "scikit-learn==1.3.0",
        "streamlit"
    ],
    python_requires=">=3.8",
)
