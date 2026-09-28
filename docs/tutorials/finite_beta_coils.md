# Finite-beta coil target

At finite current, coils must match the external vacuum jet, not the total
near-axis field. The target is

\[
(\boldsymbol B,D,H)_c=(\boldsymbol B,D,H)_{\mathrm{tot}}
-(\boldsymbol B,D,H)_p.
\]

ESSOS PR [#46](https://github.com/uwplasma/ESSOS/pull/46) contains executable
stage-two and single-stage examples. The normalized loss uses smooth
least-squares blocks for 3 field, 5 STF-gradient, and 7 STF-Hessian
components. Tests compare actual coil-shape/current gradients and
axis/`etabar` gradients with finite differences.

The formal radius must be supplied explicitly. Its asymptotic remainder
metadata and ratio to the curvature radius should be checked before accepting
a finite-beta target.
