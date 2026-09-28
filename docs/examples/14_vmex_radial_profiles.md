# VMEX radial profiles

Solves a fixed-boundary equilibrium for the section 5.1 configuration with VMEX (needs
the `vmex` extra and `PYQSC_RUN_VMEX=1`), prints $\iota(s)$, the quasisymmetry profile and
the magnetic well, and differentiates the well through the converged VMEX fixed point
({doc}`../user_guide/interfaces`). The run is deliberately coarse.

```{figure} ../_static/figures/example_14.png
:width: 75%
:alt: \iota(s) against the near-axis value (left) and the quasisymmetry residual (right).

$\iota(s)$ against the near-axis value (left) and the quasisymmetry residual (right).
```

Output:

```{literalinclude} output/14.txt
:language: text
```

```{literalinclude} ../../examples/14_vmex_radial_profiles.py
:language: python
:caption: examples/14_vmex_radial_profiles.py
```
