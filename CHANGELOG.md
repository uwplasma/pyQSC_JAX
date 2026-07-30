# Changelog

All notable changes to pyQSC_JAX will be documented here.

The project follows Semantic Versioning once the new canonical API reaches its
first stable release.

## Unreleased

### Added

- Audited compatibility and numerical baselines.
- Physics traceability and architecture decision records.
- Modern package, documentation, test, and CI scaffolding.
- Immutable general Fourier axes with normalized coefficient packing.
- JAX-native periodic differentiation, interpolation, and integration.
- Sampled Frenet geometry with explicit validity diagnostics and independent
  symmetric/asymmetric regression references.
- Converged, damped periodic sigma solve with residual, iteration,
  backtracking, stagnation, and Jacobian-conditioning reports.
- Implicit differentiation of the converged sigma equation through SOLVAX.
- Immutable first-order shape, field, field-gradient, elongation, and
  `L_grad_B` results with JIT, VMAP, JVP, and VJP validation.
- Thin ESSOS compatibility adapter preserving legacy array orientations,
  coordinate conversion, boundary generation, and plotting without depending
  on ESSOS.

### Fixed

- A no-op assignment to legacy `dofs` no longer exchanges normal and binormal
  cylindrical components.
- The legacy sigma calculation no longer assumes exactly five Newton updates.
