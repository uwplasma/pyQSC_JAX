# Differentiate radial equilibrium quantities with VMEX

`to_vmec` writes a conventional VMEC2000 input. `to_vmex_problem` instead
constructs an in-memory [VMEX](https://github.com/uwplasma/vmex) problem and
keeps the boundary and profile parameters as JAX leaves.

Install the optional dependency:

```bash
python -m pip install 'pyqsc-jax[vmex,plot]'
```

The complete executable example is:

```{literalinclude} ../../examples/14_vmex_radial_profiles.py
:language: python
```

The returned radial quantities are:

- full-mesh rotational transform `iota`;
- the native-sign `iota_vmec` profile for convention audits;
- VMEX's differentiable quasisymmetry-ratio residual at each requested
  normalized toroidal-flux surface;
- the canonical scalar magnetic well
  \((V'(0)-V'(1))/V'(0)\);
- aspect ratio, volume, magnetic energy, and thermal energy.

The call to `jax.value_and_grad` differentiates the converged fixed point by
VMEX's implicit adjoint. It does not unroll equilibrium iterations and does
not finite-difference a saved `wout`.

For an outer pyQSC_JAX optimization, keep the discrete VMEX problem fixed and
traceably remap each candidate:

```python
parameters = problem.parameters_for(candidate_solution)
objective = qsc.vmex_radial_quantities(
    problem,
    parameters,
).quasisymmetry.sum()
```

This updates the boundary Fourier arrays, toroidal flux, scalar pressure
profile, and current profile while preserving the VMEX resolution and
topology.
