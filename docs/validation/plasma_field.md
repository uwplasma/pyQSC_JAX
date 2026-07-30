# Plasma-field validation

The surface-free current and field implementation is gated by tests of:

- exact covariant/enclosed-current conversion;
- regular positive-volume radial scaling;
- vacuum, zero-current, and pressure-only limits;
- straight circular and sheared elliptical channels;
- circular toroidal finite-part and local-induction limits;
- Ampère's law, divergence, gradient symmetry, and Hessian symmetry/traces;
- full-torus periodicity and matching-length cancellation;
- angular and toroidal convergence;
- JIT/JVP and centered finite differences;
- reversible 5- and 7-component STF representations;
- independent resolved-volume Biot--Savart field comparisons with predicted
  \(a^4|\log a|\) scaling.

The Hessian uses interior-potential contact terms. Naively differentiating a
singular filament quadrature would omit those terms and is not treated as an
independent reference. See the [matched field derivation](../theory/plasma-field.md)
and `tests/physics/test_plasma_*.py`.
