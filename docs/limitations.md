# Limitations

The current computational core includes first order and the complete
finite-pressure/current r2 coefficient solve, including the total on-axis
field Hessian, Mercier terms, singular-radius diagnostics, and the r3
flux-constraint surface correction.

Known limitations under active refactor include:

- no plasma/external field-jet separation;
- pseudo-arclength continuation currently traces `etabar` branches only and
  uses a fixed arclength step.
- no multistart axis optimization or basin enumeration yet; only the exact
  affine `B2c` subproblem is eliminated.

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
`GeometryDiagnostics` reports this condition explicitly; high-level rejection
policy will be added alongside the aggregate design-criteria report.

Near-axis results are asymptotic in distance from the axis. “Surface-free”
plasma–coil separation does not mean radius-free: a formal minor radius or an
equivalent flux/current normalization is required.
