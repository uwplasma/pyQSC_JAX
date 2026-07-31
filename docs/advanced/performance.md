# Performance

Performance must be separated into compilation and execution:

- cold compile plus first execution;
- warm repeated execution;
- batched VMAP throughput;
- reverse- or forward-mode derivative cost;
- scaling with `nphi` and Fourier mode count.

The scheduled benchmark workflow records the exact dependency environment and
runs correctness checks before timing. Shared-runner timing is reported, not
used as a fragile pass/fail threshold. The main dense costs scale with the
periodic sigma Jacobian and r2 linear system; publication searches should
reuse static shapes to avoid recompilation.

For reproducible local measurements, synchronize JAX results before stopping a
timer:

```python
value = compiled_function(arguments)
value.iota.block_until_ready()
```

The release report records the benchmark command, hardware, versions, and
raw samples. No absolute performance claim is inferred from a single laptop
or CI runner.

## Refactor benchmark

The following synchronized CPU measurements were recorded on an Apple M4
(10 CPU cores, 24 GB RAM), macOS arm64, Python 3.12.13, JAX 0.11.0 with
64-bit mode, at commit `69b05da`. “Cold” clears JAX caches and includes
compilation plus execution for that static shape; it does not include Python
process or XLA-runtime startup. Warm values are medians of seven samples.
VMAP evaluates a batch of eight `etabar` values.

| `nphi` | solve cold [s] | solve warm [µs] | JVP cold [s] | JVP warm [µs] | VMAP cold [s] | VMAP warm [µs] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0.242 | 55.2 | 0.189 | 57.1 | 0.200 | 77.7 |
| 31 | 0.180 | 63.1 | 0.202 | 70.7 | 0.210 | 236.7 |
| 61 | 0.198 | 144.4 | 0.213 | 191.1 | 0.213 | 479.3 |

Compiler overhead dominates these small cold cases, while warm execution
increases with collocation size. These values are a reproducible development
snapshot, not a cross-platform speed guarantee. The exact command was:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python benchmarks/benchmark_core.py
```

All raw timing samples are stored in
`benchmarks/reports/2026-07-29-apple-m4.json`.

## VMEC boundary export

The VMEC converter is separately benchmarked because its old scalar
point-by-point root solves obscured the cost of the actual Fourier projection.
The replacement performs one JIT-compiled, vectorized Newton inversion and a
two-dimensional FFT.

| grid and spectrum | compile + execute | warm median | max \(R\) error | max \(Z\) error |
| --- | ---: | ---: | ---: | ---: |
| `nphi=61`, `ntheta=40`, `mpol=12`, `ntor=14` | 0.476 s | 5.20 ms | 3.42 µm | 2.26 µm |

The command is:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python benchmarks/benchmark_vmec_export.py
```

The reported conversion timer excludes the near-axis solve and file-system
startup. Both cold and warm calls synchronize the boundary arrays before
stopping the timer.

## VMEX implicit equilibrium

The 5.20 ms number above is only the boundary conversion. A converged VMEX
equilibrium is a separate fixed-point solve. Its cold cost includes XLA
compilation and its gradient includes an implicit adjoint linear solve.

The low-resolution live compatibility case (`ns=7`, `mpol=4`, `ntor=2`) took
23.1 s for a cold forward solve and 17.5 s for a subsequent magnetic-well
value-and-gradient on the development Apple CPU before persistent-cache
reuse. These are integration-smoke timings, not production performance
claims. Repeated optimization should reuse one `VmexProblem`, static shapes,
and VMEX's compilation cache.

The synchronized benchmark that records the dependency versions, validated
VMEX commit, case parameters, objective value, and gradient norms is:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src python benchmarks/benchmark_vmex_interface.py
```
