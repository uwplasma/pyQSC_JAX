# Limitations

The current computational core includes first order and the complete
finite-pressure/current r2 coefficient solve, including the total on-axis
field Hessian, Mercier terms, singular-radius diagnostics, and the r3
flux-constraint surface correction.

Known limitations under active refactor include:

- no magnetic-shear or \(\iota_2\) calculation yet;
- no plasma/external field-jet separation;
- no branch-aware target-transform inverse solve.

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
