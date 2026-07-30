# pyQSC_JAX

pyQSC_JAX is a differentiable JAX implementation of the near-axis expansion
for stellarator design, including a surface-free separation of total,
plasma-generated, and external vacuum field jets.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
limitations
theory/coordinates-and-conventions
theory/first-order
theory/second-order
theory/third-order
theory/field-jet
api/axis
api/first-order
api/second-order
api/third-order
api/field-jet
```

```{toctree}
:maxdepth: 2
:caption: Development

development/refactor-baseline
development/physics-traceability
adr/ADR-core-architecture
adr/ADR-solver-stack
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
```

## Project status

The immutable first-order API, complete r2 coefficient solve, r3 flux
constraint, magnetic shear, total-field Hessian, Mercier terms, and
singular-radius diagnostics are validated. Inverse solves, optimization, and
plasma/external field jets are under active development.
The `pyqsc_jax.near_axis.near_axis` import remains supported for ESSOS without
a runtime deprecation warning.

## References

The implementation is traced to primary sources, beginning with the
near-axis theory of Garren and Boozer and the direct-construction formulation
of Landreman and collaborators {cite}`garren1991existence,garren1991magnetic,landreman2018direct,landreman2019highorder`.

```{bibliography}
```
