# Installation

pyQSC_JAX requires Python 3.12 or newer.

```bash
python -m pip install pyqsc-jax
```

For development:

```bash
git clone https://github.com/uwplasma/pyQSC_JAX.git
cd pyQSC_JAX
python -m pip install -e '.[dev,docs,plot]'
```

The package depends on JAX but does not list `jaxlib` separately or configure
JAX at import time. Follow the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
to select a CPU, CUDA, or other supported backend.

Dependencies are intentionally unpinned in project metadata. Continuous
integration exercises current dependency releases, and each release report
will record the exact environment used for validation.
