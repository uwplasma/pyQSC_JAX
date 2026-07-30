# Target-transform inverse solves

## Coupled formulation

The forward first-order problem prescribes \(\bar\eta\) and solves the
periodic sigma equation for \(\sigma(\varphi)\) and \(\iota\). The inverse
problem instead fixes \(\iota_\star\) and promotes either \(\bar\eta\) or
\(I_2\) to the scalar unknown. The collocation state still has `nphi`
unknowns: one scalar parameter and `nphi - 1` sigma samples, with
\(\sigma(0)=\sigma_0\) imposed by substitution.

For `solve_for="etabar"`, the scalar state is \(u\) and

\[
\bar\eta=s_\eta\exp(u),
\qquad s_\eta\in\{-1,1\}.
\]

This prevents a Newton step from crossing the singular
\(\bar\eta=0\) surface and changing the orientation branch. The sign comes
from the seed. If no seed is supplied, `-1.0` selects the default negative
branch. A zero seed selects the positive branch from the smallest
representable positive magnitude.

For `solve_for="I2"`, the scalar state is \(I_2\) directly. In both cases the
same converged dense Newton policy and SOLVAX implicit-root rule used by the
forward sigma solve apply to the coupled inverse system.

## Local folds

After convergence, the forward sigma residual is differentiated with respect
to its ordinary state \(y=(\iota,\sigma_1,\ldots)\) and the solved parameter
\(p\). The response is obtained from

\[
\frac{\partial F}{\partial y}\frac{\mathrm dy}{\mathrm dp}
=-\frac{\partial F}{\partial p}.
\]

`response_derivative` is \(\mathrm d\iota/\mathrm dp\). Its reciprocal is the
implicit derivative of the branch-local inverse away from a fold.
`branch_fold` is true when this response is nonfinite or its magnitude is no
larger than `fold_tolerance`.

A fold does not imply that the full solution branch is singular. It means the
projection \(p(\iota)\) is locally multivalued. The direct tests contain two
different negative-\(\bar\eta\) solutions with the same transform and
opposite response slopes. Therefore a nonconverged target solve must not be
interpreted as proof that no branch exists.

## Pseudo-arclength continuation

`continue_etabar_branch` begins from two converged forward points. A secant in
physical \((\bar\eta,\iota)\) coordinates supplies the oriented tangent and
predictor. The corrector solves the complete sigma collocation system together
with

\[
\boldsymbol t\mathbin{\cdot}
\left[
(\bar\eta,\iota)-(\bar\eta,\iota)_{\mathrm{predict}}
\right]=0.
\]

The nonlinear state contains the log-magnitude of \(\bar\eta\), \(\iota\), and
all free sigma samples. Thus the corrector can pass through
\(\mathrm d\iota/\mathrm d\bar\eta=0\) without changing the selected
\(\bar\eta\) sign. A response-slope sign change marks a fold even when no
discrete point lands exactly at zero slope.

`ContinuationResult` retains every corrected immutable solution, the branch
tangents, response derivatives, and fold flags. It stops with
`status="solver_failure"` after a failed corrector or
`status="branch_zero_crossing"` before leaving the fixed-sign branch.

## Validation and acceptance

Tests cover:

- negative and positive \(\bar\eta\) sign branches;
- forward/inverse round trips for \(\bar\eta\) and finite \(I_2\);
- two distinct local \(\bar\eta\) basins at one target transform;
- full-state pseudo-arclength traversal across their intervening fold;
- independent forward-solve checks of corrected continuation points;
- r2 coefficient propagation;
- JIT and implicit JVPs against the reciprocal forward response and centered
  finite differences;
- invalid and non-scalar inputs.

Every caller must reject a result unless `root_report.converged` and
`root_report.finite` are true. Crossing a detected fold requires
pseudo-arclength continuation rather than repeated fixed-transform solves.
