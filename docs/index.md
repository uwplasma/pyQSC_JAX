# pyQSC_JAX

pyQSC_JAX constructs differentiable near-axis stellarators in JAX and
separates their on-axis total field jet into plasma-generated and external
vacuum targets without constructing a finite-radius surface.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting_started/installation
getting_started/quickstart
getting_started/choosing_a_model
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/coordinates_and_conventions
concepts/magnetic_axis
concepts/inputs_and_outputs
concepts/precision_and_units
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/first_order
theory/second_order
theory/third_order
theory/field_tensors
theory/diagnostics
theory/axis_optimization
theory/plasma_coil_separation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/first_order_qa
tutorials/first_order_qh
tutorials/second_order_finite_beta
tutorials/target_iota
tutorials/optimize_axis
tutorials/vacuum_coils
tutorials/finite_beta_coils
```

```{toctree}
:maxdepth: 2
:caption: Advanced

advanced/continuation
advanced/global_search
advanced/custom_criteria
advanced/autodiff
advanced/performance
advanced/limitations
```

```{toctree}
:maxdepth: 2
:caption: Validation and migration

validation/pyqsc_parity
validation/literature_cases
validation/plasma_field
validation/essos
migration
```

```{toctree}
:maxdepth: 2
:caption: API

api/axis
api/configurations
api/first-order
api/second-order
api/third-order
api/field-jet
api/plasma
api/continuation
api/optimization
api/axis-optimization
api/criteria
api/plotting
```

```{toctree}
:maxdepth: 2
:caption: Derivation details

theory/coordinates-and-conventions
theory/first-order
theory/second-order
theory/third-order
theory/field-jet
theory/b20-optimization
theory/criteria
theory/global-search
theory/inverse-solves
theory/plasma-current
theory/plasma-field
```

```{toctree}
:maxdepth: 2
:caption: Development

development/refactor-baseline
development/physics-traceability
development/refactor_status
adr/ADR-core-architecture
adr/ADR-solver-stack
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
release_checklist
```

## Status

The canonical API is immutable and JAX-transformable. The legacy
`pyqsc_jax.near_axis.near_axis` import remains available as a thin ESSOS
adapter. Each nonlinear and linear solve returns convergence and conditioning
evidence; callers should never accept a result solely because an object was
returned.

The physics traceability table connects each equation block to its primary
source, implementation symbol, and independent tests. The surface-free field
split always requires a formal radius or equivalent current/flux
normalization: surface-free does not mean radius-free.

## References

The implementation begins with the near-axis construction of Garren and
Boozer and the direct cylindrical-coordinate formulations of Landreman and
collaborators
{cite}`garren1991existence,garren1991magnetic,landreman2018direct,landreman2019highorder`.

```{bibliography}
```
