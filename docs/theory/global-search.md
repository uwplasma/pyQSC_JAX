# Branch-aware axis search

## Objective and exact scalar elimination

`search_axis` minimizes the full weighted, mean-free \(B_{20}\) residual on
the toroidal grid:

\[
r_j=\sqrt{\frac{w_j}{\sum_kw_k}}\,
\frac{B_{20,j}-\langle B_{20}\rangle_w}{B_0}.
\]

It does not use a few hand-selected collocation points. At every axis
evaluation, \(B_{2c}\) is eliminated analytically before the residual is
formed. Target-transform problems use `solve_for="etabar"` or
`solve_for="I2"`, so the requested transform and the full sigma collocation
state are solved together on the selected branch.

## Reproducible global-to-local workflow

The implemented workflow is:

1. include the supplied axis and any explicit branch-specific seeds;
2. fill the bounded coefficient box with a deterministic Halton sequence;
3. evaluate the coarse population with `jax.vmap`, replacing invalid
   geometry or unconverged solves by an infinite score without contaminating
   the batch with NaNs;
4. refine the best starts with a damped Gauss–Newton/Levenberg–Marquardt
   iteration using the full JAX residual Jacobian;
5. cluster endpoints in normalized bounded-coordinate space;
6. apply hard geometry, solver, conditioning, and optional `Criteria` checks;
7. select lexicographically by primary tolerance and then by the requested
   secondary selector; and
8. independently rebuild the selected physical candidate at the requested
   toroidal resolutions.

Every optimized variable is represented internally by a hyperbolic-tangent
map. Bounds are therefore never imposed by clipping a trial step. The width of
each coefficient interval supplies mode-aware variable scaling; callers should
tighten high-mode bounds as the Fourier order increases.

`continue_axis_search` provides staged Fourier continuation. Each stage may
increase the retained mode count and `nphi`; all common Fourier coefficients
from the prior best basin warm-start the next stage. This is preferable to
activating many weakly resolved high modes at once.

The built-in secondary selectors rank feasible basins by singular radius,
minimum first- or second-derivative scale length, maximum elongation, a
fourth-order axis Sobolev norm, distance from a target axis length, or minimum
normalized criteria margin. They are not silently mixed into the primary
quasisymmetry residual with arbitrary weights.

## Status and global claims

The result uses one of:

- `verified_zero`;
- `best_found`;
- `no_feasible_candidate`;
- `branch_fold`;
- `ill_conditioned`;
- `solver_failure`; or
- `verification_failure`.

`verified_zero` has one deliberately narrow meaning. The nonconstant
\(B_{20}\) residual is nonnegative, so a candidate whose weighted and maximum
residuals reach zero within the requested absolute, resolution, and spectral
tolerances attains the known global lower bound of that primary residual. The
flag does not certify that the secondary selector is globally optimal.

Any nonzero result is `best_found` and reports its search budget and number of
distinct basins. It carries no global-minimum claim. A small coarse-grid value
that fails doubled-grid, maximum-error, or spectral-tail checks is
`verification_failure`.

## High-frequency safeguards

The search exposes coefficient bounds, bounded-coordinate scaling, staged
mode activation, a Sobolev selector, the complete nonzero Fourier spectrum,
its \(L^1\) certificate, tail ratio, and doubled/quadrupled-grid changes.
These controls are important because high-order axis optimization is
spectrally ill-conditioned: unresolved coefficients can move error between
collocation points without improving the continuous configuration.
