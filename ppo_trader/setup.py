from setuptools import setup, find_packages

setup(
    name="ppo_trader",
    version="0.3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pandas",
    ],
    extras_require={
        "mamba": ["mamba-ssm"],
    },
    python_requires=">=3.9",
)
