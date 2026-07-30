# Third order

Third order imposes the toroidal-flux constraint needed for pyQSC-compatible
boundary harmonics and exposes an explicit magnetic-shear calculation.
`order="r3"` produces first- and third-poloidal-harmonic coefficients and
two consistency residuals. `solve_magnetic_shear` is separate because
`B31c` is an independent input.

The [third-order and shear derivation](third-order.md) records equations,
supported sign conventions, and upstream parity cases. Validation is in
`tests/physics/test_third_order.py` and `tests/physics/test_shear.py`.
