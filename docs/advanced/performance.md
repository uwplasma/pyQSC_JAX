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
