# Installation

pyQSC_JAX needs Python 3.12 or newer and JAX.

```bash
python -m pip install pyqsc-jax            # core
python -m pip install 'pyqsc-jax[plot]'    # + Matplotlib helpers
python -m pip install 'pyqsc-jax[vmex]'    # + differentiable VMEX equilibria
```

From a clone, for development:

```bash
git clone https://github.com/uwplasma/pyQSC_JAX.git
cd pyQSC_JAX
python -m pip install -e '.[dev,docs,plot]'
```

The package depends on `jax` and on [SOLVAX](https://github.com/uwplasma/SOLVAX) for the
implicit root and linear solves. It does not choose a JAX backend; install the CPU, CUDA
or other build from the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## 64-bit arithmetic

All computations require 64-bit floats. The package never changes JAX configuration on
import, so enable them yourself, before any array is created:

```bash
export JAX_ENABLE_X64=1
```

or, at the top of a script,

```python
import jax
jax.config.update("jax_enable_x64", True)
```

In 32-bit mode the spectral derivatives, the second-order linear system and the field
Hessian lose most of their digits; none of the numbers in this documentation apply.

## Units

SI throughout: lengths in metres, fields in tesla, `p2` in Pa/m², `I2` in T/m (the
pyQSC covariant-current normalization), `etabar` in 1/m, `B2c`, `B2s` and `B20` in T/m².
Angles and $\iota$ are dimensionless.
