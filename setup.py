from setuptools import setup, find_packages

setup(
    name="pyqsc_jax",
    version="0.1.0",
    author="Agastya Rathee",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
