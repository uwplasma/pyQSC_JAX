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
api/axis
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

The immutable canonical API is under active development. The
`pyqsc_jax.near_axis.near_axis` import remains supported for ESSOS without a
runtime deprecation warning.

## References

The implementation is traced to primary sources, beginning with the
near-axis theory of Garren and Boozer and the direct-construction formulation
of Landreman and collaborators {cite}`garren1991existence,garren1991magnetic,landreman2018direct,landreman2019highorder`.

```{bibliography}
```
