# Refactor baseline

Date: 2026-07-29

This audit establishes the starting point for the complete pyQSC_JAX refactor.
It is descriptive, not an endorsement of the current architecture.

## Repository and permission status

| Repository | Audited revision | Access | Local status |
| --- | --- | --- | --- |
| `uwplasma/pyQSC_JAX` | `8b5d8ea218a624071765bfbf62d8d8e0c7acb90c` | admin/push | clean before audit; feature branch created |
| `landreman/pyQSC` | `cd75359ea47548d5db7ccb458c100085c04ba1bc` | read-only reference | clean |
| `uwplasma/ESSOS` | `932997f198964ccfffea898e25acd73600adbf3a` | admin/push | existing worktree has unrelated generated output |
| `uwplasma/SOLVAX` | `bf219c79a40a80b2c3115b749bf670b9f5de3cb5` | admin/push | existing worktree has unrelated changes |

GitHub authentication is active for `rogeriojorge`. Any later ESSOS or SOLVAX
work must use an isolated worktree so the pre-existing local changes remain
untouched.

## Current pyQSC_JAX inventory

The package consists of an empty `pyqsc_jax/__init__.py`, a 505-line
`pyqsc_jax/near_axis.py`, `setup.py`, a title-only README, and no tests or CI.
The monolithic mutable `near_axis` class currently combines:

- stellarator-symmetric axis Fourier evaluation using only `rc` and `zs`;
- Frenet geometry and topology;
- the first-order sigma equation;
- field and gradient evaluation;
- coordinate conversion and boundary generation;
- mutable optimization degrees of freedom.

`order`, `B2c`, and `p2` are accepted but do not activate second- or third-order
physics. The sigma equation is advanced by exactly five Newton steps with no
convergence criterion or report. Methods are JIT-compiled with mutable `self`
as a static argument. Package metadata constrains both `jax` and `jaxlib`.

The unmerged `origin/en/essos_bridge_fix` branch adds plotting and VTK export,
but also imports ESSOS from pyQSC_JAX and commits generated artifacts. Required
behavior will be reimplemented without reversing the dependency direction.

## Numerical baseline

For

```python
near_axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    etabar=-0.9,
    nfp=3,
    nphi=31,
)
```

with JAX 64-bit arithmetic, current pyQSC_JAX and upstream pyQSC agree to
floating-point precision:

| Quantity | Baseline |
| --- | ---: |
| `iota` | 0.41830690943386617 |
| `axis_length` | 6.340238817434161 |
| minimum curvature | 0.5956163516982903 |
| maximum curvature | 1.3060121594235536 |
| minimum torsion | -2.272197039914583 |
| maximum torsion | 0.6057564303422814 |
| maximum absolute sigma | 0.9920706836908618 |
| `B_axis.shape` | `(3, 31)` |
| `grad_B_axis.shape` | `(3, 3, 31)` |

The audit environment used unpinned JAX/JAXLIB 0.11.0, NumPy 2.5.1,
SciPy 1.18.0, and Python 3.12.

## Confirmed defect

Assigning `field.dofs = field.dofs` unpacks the normal and binormal cylindrical
components in a different order than the constructor. The no-op round trip
changes the frame by up to:

- normal: `0.5460213389358983`;
- binormal: `1.1249025277689044`.

The strict expected-failure regression test must become a passing test when the
compatibility adapter is corrected.

## Compatibility tiers

### Tier 1: ESSOS, mandatory throughout

ESSOS PR `uwplasma/ESSOS#34` established the import:

```python
from pyqsc_jax.near_axis import near_axis
```

ESSOS constructs the class from `rc`, `zs`, `etabar`, `nfp`, and `nphi`.
Production objectives currently consume `R0`, `Z0`, `phi`, `B_axis`,
`grad_B_axis`, and `iota`; optimization additionally consumes `rc`, `zs`,
`etabar`, `x`, `dofs`, `B0`, `sigma0`, `I2`, `nphi`, `spsi`, `sG`, and `nfp`.
Examples also rely on boundary conversion and plotting.

### Tier 2: upstream pyQSC semantic parity, final acceptance

DESC, SIMSOPT, and broader pyQSC workflows use a larger object protocol,
including general `rc`, `rs`, `zc`, `zs`, `Bbar`, `lasym`, pressure/current
coefficients, interpolation splines, second- and third-order boundary data, and
named configurations. This tier is staged after the immutable first-order core
but remains part of the final definition of done.

## Upstream test audit

At pyQSC commit `cd75359`, 31 tests pass in 22.80 seconds when
`test_to_vmec.py` is excluded because its optional MPI/VMEC runtime is absent.
The tests cover:

- Newton convergence, Fourier differentiation, interpolation, and utilities;
- axis geometry, helicity, sigma, iota, field gradient, and scale lengths;
- comparison to independent Fortran netCDF outputs;
- complete second-order coefficients, `B20`, Mercier terms, and singular radius;
- third-order coefficients, shear, and boundary Fourier conversion;
- named paper configurations and independent curvature/torsion checks.

Reference data imported later must be reduced to small arrays, retain BSD-2-Clause
attribution, record this upstream commit and checksums, and never download during
tests.

## Initial risks and gates

1. Frenet coordinates require nonvanishing axis curvature; invalid axes need an
   explicit diagnostic rather than silent NaNs.
2. The target-iota inverse can cross folds where
   `d(iota)/d(etabar) = 0`; scalar inversion alone is not globally valid.
3. The second-order operator can be ill-conditioned near `iota_N = 0`.
4. A small collocation residual does not establish a small dense-grid `B20`
   variation.
5. The manuscript plasma Hessian is gated on independent volume-current
   Biot–Savart validation and Maxwell identities.
6. “Surface-free” still requires a formal radius or equivalent flux/current
   normalization.
7. Nonzero optimization results are “best found,” never proofs of a global
   minimum.
