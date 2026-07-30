# First order

First order solves the periodic sigma equation for cross-section orientation
and rotational transform, then constructs the elliptical surface coefficients
and on-axis field gradient. The nonlinear solve is damped, convergence-tested,
and differentiated at the converged root through the implicit-function
theorem.

The complete equations, sign conventions, failure modes, implementation
symbols, and reference value are in the
[first-order derivation](first-order.md). Direct tests live in
`tests/physics/test_first_order.py`; AD checks are in
`tests/numerics/test_first_order_autodiff.py`.
