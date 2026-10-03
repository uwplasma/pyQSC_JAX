from setuptools import setup, find_packages

setup(
    name="pyqsc_jax",
    version="0.1.0",
    author="Uw_Plasma",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    extras_require={"essos": ["essos>=0.19.4"]},
    python_requires=">=3.9",
)
