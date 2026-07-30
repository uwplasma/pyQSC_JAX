# Limitations

The current computational core includes first order and the complete
finite-pressure/current r2 coefficient solve, including the total on-axis
field Hessian, Mercier terms, singular-radius diagnostics, and the r3
flux-constraint surface correction.

Known limitations under active refactor include:

- the plasma Hessian and complete 3+5+7 external vacuum jet are not yet
  exposed; the positive-volume source, matched on-axis plasma field, local
  plasma gradient, and 3+5 external target are implemented.
- pseudo-arclength continuation currently traces `etabar` branches only and
  uses a fixed arclength step.

The bounded axis search is deterministic and enumerates distinct locally
refined basins, but a finite multistart budget is not an exhaustive proof over
a high-dimensional coefficient box. Only `verified_zero` certifies the known
zero lower bound of the nonnegative primary \(B_{20}\) residual. Built-in
secondary selectors currently rank the distinct primary-refined basins; they
do not carry a separate global certificate.

`Criteria.from_curvo_2025` is a configurable reproduction of one published
screening profile. Its thresholds are not hard-coded package-wide acceptance
requirements. In particular, `r_singularity` is the first singularity of the
truncated near-axis coordinate map; it does not guarantee nested,
nonintersecting finite-radius flux surfaces.

Target-transform solves are branch-local. The sign of the `etabar` seed is
preserved and the seed magnitude selects a Newton basin. Multiple local
solutions can have the same transform. `response_derivative` and `branch_fold`
detect local loss of invertibility. `continue_etabar_branch` crosses these
folds using the full sigma collocation state and an augmented arclength
condition.

The magnetic-shear calculation currently implements the standard-MHS
specialization with \(B_{31s}=0\), \(I_4=0\), and \(s_G=s_\psi=1\). Other sign
conventions are rejected explicitly pending a traced derivation.

The sigma equation is solved to configurable residual and step tolerances, but
a result object is still returned after nonconvergence so callers can inspect
the structured report. Production workflows must reject
`root_report.converged == False`.

Frenet coordinates are invalid when the magnetic-axis curvature vanishes.
`GeometryDiagnostics` reports this condition explicitly. The published Curvo
profile does not add a minimum-curvature threshold, but callers can reject
invalid geometry independently.

Near-axis results are asymptotic in distance from the axis. “Surface-free”
plasma–coil separation does not mean radius-free: a formal minor radius or an
equivalent flux/current normalization is required.
