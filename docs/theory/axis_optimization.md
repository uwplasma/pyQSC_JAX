# Axis optimization

The primary search residual is the weighted, mean-free \(B_{20}\) array.
For fixed axis and other physical inputs, `B2c` is eliminated analytically.
Bounded deterministic exploration feeds damped local least-squares solves;
endpoints are clustered into distinct basins and rebuilt on independent grids.

Only `verified_zero` carries a global statement: it reaches the known zero
lower bound of the nonnegative primary residual within requested tolerances.
A nonzero `best_found` result is not a global-minimum proof. Secondary
selectors rank already refined basins and carry no separate certificate.

See [branch-aware global search](global-search.md) and
[\(B_{20}\) diagnostics](b20-optimization.md) for equations, status
semantics, high-mode controls, and verification.
