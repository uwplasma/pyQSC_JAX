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

## Plasma-dominant regression cases

`plasma_dominant_channel` remains the circular analytic regression: it makes
the current normalization transparent and produces the same 33.2574% fraction
at every angle.

The README and example use `plasma_stellarator`, a gently shaped
two-field-period case with `I2=4.2`, `p2=-1e5`, `B0=1`, and formal radius
0.2 m. It retains the \(>30\%\) plasma contribution while making the split
angle dependent:

| quantity | value |
| --- | ---: |
| minimum/mean/maximum \(\lvert B_p\rvert/\lvert B_\mathrm{tot}\rvert\) | 0.326967 / 0.332552 / 0.338719 |
| plasma \(\lvert B\rvert\) peak-to-peak / mean | 0.03534 |
| external \(\lvert B\rvert\) peak-to-peak / mean | 0.02663 |
| enclosed toroidal current | 0.84 MA |
| formal radius | 0.2 m |
| computed singular radius | 0.3351 m |
| formal/singular radius | 0.597 |
| estimated field remainder | 0.0351 T |

The minimum fraction, nonzero angular variation, finite current, and radius
margin are regression assertions. The publication plot shows total, plasma,
and external Frenet components; plotting only their norms would hide both the
constant-\(B_0\) total-field construction and the exact cancellation of
transverse plasma/external components.

This is a field-split demonstration rather than a device-quality optimum. Its
role is to ensure that the plasma contribution cannot silently collapse to a
negligible plotting scale while retaining explicit asymptotic-error and
singular-radius metadata.
