# Export a fixed-boundary VMEC input

The exporter does not require VMEC. It constructs the near-axis surface,
inverts cylindrical toroidal angle on a uniform grid, projects all four
boundary coefficient families with a two-dimensional FFT, writes a
deterministic `&INDATA` file, and returns accuracy and timing diagnostics.
If the requested surface does not admit a converged cylindrical-angle
inversion, `to_vmec` raises before creating the file.

```{literalinclude} ../../examples/13_vmec_export.py
:language: python
```

For an asymptotic transform check, start at a small radius. Increasing the
radius tests finite-radius behavior of the truncated near-axis surface as well
as the exporter. The measured convergence is documented in
{doc}`../validation/vmec`.
