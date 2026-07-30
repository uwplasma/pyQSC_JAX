# Precision and units

The code uses SI units:

- length in metres;
- magnetic field in tesla;
- pressure in pascals;
- current in amperes when explicitly converted;
- `I2` in the pyQSC covariant-current normalization.

Angles and transform are dimensionless. Fourier axis coefficients have units
of length; `etabar` has inverse-length units; `B2c`, `B2s`, and `B20` have
T/m\(^2\).

Enable JAX 64-bit arithmetic in the process environment before import:

```bash
JAX_ENABLE_X64=true python calculation.py
```

The library does not make that global choice for callers. Reference parity,
small Maxwell residuals, and high-order spectral differentiation require
64-bit mode. In 32-bit mode, use problem-appropriate tolerances and do not
compare against the documented double-precision residuals.

Resolution is part of the numerical model. Recompute accepted r2/r3 and
plasma-field candidates on doubled grids; inspect Fourier tails and solver
condition numbers rather than treating `nphi` as a cosmetic plotting choice.
