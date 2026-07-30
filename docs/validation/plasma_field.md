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

## Pressure-only stellarator showcase

The README and every finite-beta example use `plasma_stellarator`, the public
[Wisconsin stellarator-database configuration 52521](https://stellarator.physics.wisc.edu/app/plot/52521).
It has exactly `I2=0`, finite `p2=-28248.188`, four field periods, and a
strongly nonplanar magnetic axis. At `nphi=121`, its RMS torsion is
\(0.978864\ \mathrm{m}^{-1}\), so this case cannot silently regress to a
planar tokamak axis. It passes every configurable Curvo criterion with the
stricter \(\lvert\iota\rvert\ge0.4\).

| quantity | value |
| --- | ---: |
| \(I_2\) | exactly 0 |
| \(p_2\) | \(-2.82482\times10^4\ \mathrm{Pa/m^2}\) |
| minimum/mean/maximum \(\lvert B_p\rvert/\lvert B_\mathrm{tot}\rvert\) | 0.00176059 / 0.00188960 / 0.00202583 |
| plasma \(\lvert B\rvert\) peak-to-peak / mean | 0.140368 |
| enclosed toroidal current | exactly 0 A |
| \(\lvert\iota\rvert\) | 2.809271 |
| RMS axis torsion | \(0.978864\ \mathrm{m}^{-1}\) |
| formal radius | 0.15 m |
| computed singular radius | 0.391608 m |
| formal/singular radius | 0.383 |
| estimated Hessian remainder | \(5.54\times10^{-3}\ \mathrm{T/m^2}\) |

The zero current, finite pressure, nonzero torsion, nonzero plasma field,
angular variation, and radius margin are regression assertions. The
publication plot shows total, plasma, and external Frenet components;
plotting only their norms would hide both the constant-\(B_0\) total-field
construction and the cancellation of transverse plasma/external components.

The pressure-only plasma contribution is about \(0.19\%\), not 30%. A scan
over screened database configurations and pressure multipliers found that
forcing a current-visible 30% fraction with \(I_2=0\) is incompatible with
the regular, criterion-passing near-axis examples considered here. The
earlier 30% showcase obtained its scale from finite \(I_2\) and appeared
tokamak-like, so it has been removed. `plasma_dominant_channel` is retained
only as a circular analytic current-normalization regression; it is not a
showcase stellarator or a recommended design.
