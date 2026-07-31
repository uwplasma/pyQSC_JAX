# VMEC validation

## Conversion algorithm

`uniform_cylindrical_surface` solves the cylindrical-toroidal angle inversion
for the entire poloidal/toroidal grid with six vectorized JAX Newton
steps. Its derivative is obtained by JVP, so no finite-difference step is
selected. `vmec_boundary` reports the maximum residual, a dtype-aware default
tolerance of 100 machine epsilons, and a convergence flag. `to_vmec` raises
before writing if that flag is false; this prevents a large, non-star-shaped
or otherwise failed surface inversion from looking like a valid VMEC input.
The tolerance can be tightened explicitly.

After the angle solve, `vmec_boundary` computes `RBC`, `RBS`, `ZBC`, and
`ZBS` simultaneously with a two-dimensional FFT. This replaces the legacy
scalar dense-root solve at every surface point and the direct Fourier
quadrature over every retained mode.

At `nphi=61`, `ntheta=40`, `mpol=12`, and `ntor=14`, the new surface agrees
with the independent legacy root-based conversion to \(1.6\times10^{-14}\) m.
At radius 0.03 m, its retained-spectrum reconstruction errors are
\(3.42\times10^{-6}\) m in \(R\) and \(2.26\times10^{-6}\) m in \(Z\);
the maximum toroidal-angle residual is \(2.22\times10^{-16}\) rad.

## Local equilibrium check

The checked-in validation case was run with local VMEC2000 9.0 on macOS
arm64. It uses

- `rc=[1.0, 0.045]`, `zs=[0.0, -0.045]`, `nfp=3`;
- `etabar=-0.9`, `B0=1`, `nphi=121`, order `r2`;
- export radius \(r=0.0025\) m;
- `ntheta=32`, `mpol=6`, `ntor=6`, `ns=31`.

| Quantity | Near axis | VMEC |
| --- | ---: | ---: |
| on-axis \(\iota\) | 0.4183069102 | 0.4185430697 |
| relative difference | — | 0.05646% |
| `fsqr` | — | \(7.59\times10^{-11}\) |
| `fsqz` | — | \(4.21\times10^{-11}\) |
| `fsql` | — | \(2.40\times10^{-11}\) |

The fixed
[`wout_qa_r0025.nc`](../../tests/reference/vmec/wout_qa_r0025.nc)
and its [manifest](../../tests/reference/vmec/manifest.json) are read on every
test run. The test checks the checksum, normal termination, force residuals,
and on-axis transform. Set
`PYQSC_VMEC_EXECUTABLE=/path/to/xvmec` to enable the second integration test,
which regenerates the input and reruns VMEC in a temporary directory.

## Finite-pressure, zero-current database check

A second opt-in local test exercises the user-facing finite-beta regime rather
than a finite-current channel. It uses Wisconsin stellarator-database QA
[ID 139524](https://stellarator.physics.wisc.edu/app/plot/139524), which has
finite \(p_2=-2.32744\times10^5\ \mathrm{Pa/m^2}\), exactly \(I_2=0\),
\(|\iota|=0.354802\), and RMS axis torsion
\(1.189\ \mathrm{m}^{-1}\). The export uses \(r=0.0015\) m,
`nphi=241`, `ntheta=40`, `mpol=8`, `ntor=8`, and radial stages
`ns=(31, 61)`.

| Quantity | Near axis | VMEC |
| --- | ---: | ---: |
| on-axis \(\iota\) | \(-0.35480218\) | \(-0.35472615\) |
| relative difference | — | 0.02143% |
| on-axis pressure | 0.523673 Pa | 0.523673 Pa |
| enclosed current | exactly 0 A | exactly 0 A input |
| maximum force residual | — | \(9.97\times10^{-12}\) |

This small radius is intentional: the transform comparison is an asymptotic
on-axis validation, while finite-beta radial behavior at practical boundary
radii is exercised by the VMEX integration.

## Radius convergence

The on-axis transform comparison is asymptotic. The same low-resolution VMEC
setup gives:

| radius [m] | VMEC on-axis \(\iota\) | relative difference |
| ---: | ---: | ---: |
| 0.0025 | 0.41854307 | 0.0565% |
| 0.0050 | 0.41922558 | 0.2196% |
| 0.0100 | 0.42224025 | 0.9403% |
| 0.0200 | 0.43955604 | 5.0798% |
| 0.0300 | 0.49070165 | 17.3066% |

The first three points show the expected approximately \(O(r^2)\) error. The
larger-radius results are intentionally retained: agreement at a small radius
does not imply that a truncated near-axis boundary is accurate arbitrarily far
from the axis.

## Performance

On the Apple M4 development machine, conversion at the higher
`40 × 61`, `mpol=12`, `ntor=14` resolution took 0.476 s including JAX
compilation and 5.20 ms after compilation. These are synchronized local
measurements, not cross-platform pass/fail promises. The unit test uses a
generous 0.5 s warm-call ceiling to catch a return to per-point root solving.
Raw data and the runnable command are in
`benchmarks/reports/2026-07-30-b20-vmec-apple-m4.json` and
`benchmarks/benchmark_vmec_export.py`.
