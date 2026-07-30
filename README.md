# pyQSC_JAX

Differentiable near-axis stellarator construction and plasma–coil field jets in
JAX.

> **Development status:** the current release preserves the ESSOS first-order
> interface while an immutable, fully validated pyQSC-compatible core is built
> on the `refactor/pyqsc-jax-complete` branch.

## Install

```bash
python -m pip install pyqsc-jax
```

For a source checkout:

```bash
python -m pip install -e .
```

No JAX backend policy is imposed by the package. Install the JAX build suitable
for your accelerator and platform.

## Current quickstart

```python
from pyqsc_jax.near_axis import near_axis

field = near_axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
)

print("iota:", field.iota)
print("axis length:", field.axis_length)
print("minimum L_grad_B:", field.L_grad_B.min())
```

The import above is the supported ESSOS compatibility API. The refactor is
introducing the canonical immutable form:

```python
import pyqsc_jax as qsc

configuration = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    order="r2",
)
```

That API will become the primary quickstart once its first- and second-order
validation gates are green.

The lower-level immutable axis API is already available:

```python
import pyqsc_jax as qsc
from pyqsc_jax.geometry import compute_axis_geometry

axis = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
geometry = compute_axis_geometry(axis, nphi=31)

print("axis length:", geometry.axis_length)
print("minimum curvature:", geometry.diagnostics.minimum_curvature)
```

## Scope

The completed package will provide:

- general Fourier magnetic axes and QA/QH topology;
- converged differentiable first-order sigma solves;
- complete finite-pressure/current second order and pyQSC third-order parity;
- total, plasma, and external field/gradient/Hessian jets on the axis;
- target-transform inverse solves with continuation and fold detection;
- dense-grid `B20` diagnostics and multistart axis optimization;
- cited stellarator screening criteria with explicit margins;
- an unchanged ESSOS compatibility adapter.

See the [draft refactor PR](https://github.com/uwplasma/pyQSC_JAX/pull/2),
[architecture decisions](docs/adr/), and
[physics traceability](docs/development/physics-traceability.md).

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). pyQSC_JAX is MIT
licensed. Adapted pyQSC source and reference data retain BSD-2-Clause
attribution in [NOTICE](NOTICE).
