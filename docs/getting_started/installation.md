# Installation

pyQSC_JAX requires Python 3.12 or newer.

```bash
python -m pip install pyqsc-jax
```

Install optional plotting, VMEX equilibrium, or development tools explicitly:

```bash
python -m pip install 'pyqsc-jax[plot]'
python -m pip install 'pyqsc-jax[plot,vmex]'
git clone https://github.com/uwplasma/pyQSC_JAX.git
cd pyQSC_JAX
python -m pip install -e '.[dev,docs,plot,vmex]'
```

The `vmex` extra enables differentiable fixed-boundary radial quantities. To
test the current upstream development head directly:

```bash
python -m pip install 'git+https://github.com/uwplasma/vmex.git'
```

The project declares `jax`, not `jaxlib`, and never changes JAX configuration
at import time. Choose the CPU, CUDA, or other backend using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Production calculations should enable 64-bit arithmetic before importing JAX:

```bash
export JAX_ENABLE_X64=true
```

Dependencies are intentionally unpinned in package metadata. Release reports
record the concrete environment used for validation.
