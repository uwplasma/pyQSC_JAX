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

## Plasma-dominant regression case

The public `plasma_dominant_channel` case is a circular finite-pressure/current
channel with `I2=4.2`, `p2=-1e5`, `B0=1`, and formal radius 0.2 m. It is used
to keep the plasma/external split visually and numerically consequential:

| quantity | value |
| --- | ---: |
| minimum/mean/maximum \(\lvert B_p\rvert/\lvert B_\mathrm{tot}\rvert\) | 0.332574 / 0.332574 / 0.332574 |
| enclosed toroidal current | 0.84 MA |
| formal radius | 0.2 m |
| computed singular radius | 0.4034 m |
| formal/singular radius | 0.496 |
| estimated field remainder | 0.0351 T |

The \(>30\%\) fraction, finite current, and radius margin are regression
assertions. This case is intentionally transparent rather than a claim of an
optimized stellarator: its role is to validate that the plasma contribution
cannot silently collapse to a negligible plotting scale.
