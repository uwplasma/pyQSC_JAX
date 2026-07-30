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
two-field-period case with `I2=1.55`, `p2=-1e3`, `B0=1`, and formal radius
0.45 m. Unlike the earlier high-current demonstration, this pressure/current
balance passes every configurable Curvo criterion with the stricter
\(\lvert\iota\rvert\ge0.4\). It retains the \(>30\%\) plasma contribution
while making the split angle dependent:

| quantity | value |
| --- | ---: |
| minimum/mean/maximum \(\lvert B_p\rvert/\lvert B_\mathrm{tot}\rvert\) | 0.329933 / 0.337868 / 0.344899 |
| plasma \(\lvert B\rvert\) peak-to-peak / mean | 0.04429 |
| external \(\lvert B\rvert\) peak-to-peak / mean | 0.01042 |
| enclosed toroidal current | 1.569375 MA |
| \(\lvert\iota\rvert\) | 0.730286 |
| formal radius | 0.45 m |
| computed singular radius | 0.9916 m |
| formal/singular radius | 0.454 |
| estimated field remainder | 0.2286 T |

The minimum fraction, nonzero angular variation, finite current, and radius
margin are regression assertions. The publication plot shows total, plasma,
and external Frenet components; plotting only their norms would hide both the
constant-\(B_0\) total-field construction and the exact cancellation of
transverse plasma/external components.

The underlying near-axis stellarator passes the named device-screening
profile; the \(a=0.45\) field split is deliberately a current-visible
demonstration, not a high-accuracy finite-radius claim. Its relatively large
0.2286 T remainder estimate is displayed here rather than hidden. Production
use should reduce the formal radius and perform the documented asymptotic
radius convergence, which will also reduce the plasma fraction unless the
current/geometry is re-optimized. The regression's role is to ensure that the
plasma contribution cannot silently collapse to a negligible plotting scale
while retaining explicit asymptotic-error and singular-radius metadata.
