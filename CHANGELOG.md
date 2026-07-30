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
- Complete finite-pressure/current r2 coefficient system, including
  `X20`/`Y20`, second harmonics, `beta_1s`, `G2`, `B20`, derivatives,
  untwisted boundary data, and direct `B20` diagnostics.
- Implicit differentiation and residual/conditioning reports for the coupled
  dense second-order solve.
- Independent four-equation r2 residual checks and upstream regression cases
  spanning vacuum QA, finite pressure/current, and QH topology.
- Regular-coordinate total-field value, gradient, and Hessian on the axis,
  including coordinate-map conditioning and Maxwell-identity diagnostics.
- Mercier magnetic-well and geodesic terms matching vacuum and finite-pressure
  pyQSC references.
- JIT/JVP, resolution, upstream Hessian, full vacuum-symmetry, divergence, and
  derivative-of-divergence validation for the total field jet.
- Singular-radius diagnostics derived directly from the regular-coordinate
  Jacobian determinant, with vectorized global angular seeding, Newton
  refinement, residual reports, and pyQSC QA/QH/finite-current parity.
- Differentiable r3 flux-constraint surface corrections, untwisted boundary
  coefficients, and two independent consistency checks with upstream pyQSC
  QA, finite-pressure/current, and QH parity.
- Standard-MHS magnetic shear with periodic and secular integrating-factor
  branches, explicit `B31c`, immutable diagnostics, legacy adapter support,
  and upstream QA/QH/asymmetric parity.

### Fixed

- A no-op assignment to legacy `dofs` no longer exchanges normal and binormal
  cylindrical components.
- The legacy sigma calculation no longer assumes exactly five Newton updates.
- Canonical field gradients now use explicit
  `(sample, field component, derivative direction)` ordering; the ESSOS
  adapter retains its historical layout.
