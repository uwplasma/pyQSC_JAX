# ADR: immutable functional core with compatibility adapters

- Status: accepted
- Date: 2026-07-29

## Context

The current mutable `near_axis` class mixes inputs, derived arrays, nonlinear
solves, coordinate conversion, field evaluation, plotting, and optimizer state.
Passing the mutable object as a static JIT argument makes cache behavior and
automatic differentiation fragile. ESSOS nevertheless requires the historical
class and mutable degree-of-freedom interface.

## Decision

Build a canonical immutable API from small frozen dataclasses registered as JAX
pytrees and pure functions. Keep array-valued physics data as pytree leaves and
small topology/resolution choices as static metadata. Ordinary dataclasses are
preferred initially; Equinox will be added only if it removes demonstrated
complexity.

The public API will provide `Axis`, `Qsc`, and `solve`. The legacy
`pyqsc_jax.near_axis.near_axis` class will be a thin adapter that reconstructs
the immutable solution when its mutable degrees of freedom change.

Modules will follow physics and ownership boundaries, beginning with models,
spectral operators, axis geometry, solvers, and first order. Empty speculative
modules and generic `utils.py` are prohibited.

## Consequences

- JIT, VMAP, JVP, and VJP operate on explicit array arguments.
- Solver reports and invalid-geometry diagnostics become part of results.
- Compatibility mutation remains isolated and testable.
- General asymmetric axis coefficients can be supported without duplicating
  geometry formulas.
- The adapter may allocate a new solution after mutation; this cost is
  acceptable for compatibility and must not constrain the core.
