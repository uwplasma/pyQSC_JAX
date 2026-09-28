# Second order

Second order supplies the complete finite-pressure/current coefficient system,
including the coupled periodic `(X20, Y20)` solve, `G2`, `beta_1s`, and
`B20`. The dense linear system is solved once for the primal result and
differentiated implicitly through SOLVAX.

See the [complete second-order derivation](second-order.md) for the ordered
solution steps, residual equations, condition report, and pyQSC references.
The implementation is `pyqsc_jax.second_order`; independent residual and AD
tests are in `tests/physics/test_second_order.py` and
`tests/numerics/test_second_order_autodiff.py`.
