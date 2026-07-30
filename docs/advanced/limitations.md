# Limitations

- Near-axis results are asymptotic and do not prove a finite-radius
  equilibrium or particle confinement.
- Frenet coordinates fail where axis curvature vanishes.
- Target-transform inversions are branch-local; folds require continuation.
- Pseudo-arclength continuation currently traces `etabar` with a fixed step.
- Magnetic shear currently supports the documented standard-MHS sign
  specialization.
- A finite multistart budget cannot certify a nonzero global minimum.
- `r_singularity` diagnoses the truncated coordinate map, not equilibrium
  existence.
- Plasma-field and Hessian remainder fields are asymptotic scales, not
  rigorous bounds.
- The optional VMEX bridge covers differentiable fixed-boundary equilibria.
  It does not differentiate a reconverged free-boundary NESTOR root.
- Traceable VMEX quasisymmetry profiles currently require stellarator
  symmetry; use `qs_surfaces=()` for an asymmetric equilibrium without QS.
- VMEX magnetic well is an endpoint scalar, not a radial profile.
- “Surface-free” plasma/coil separation still requires a positive formal
  radius or equivalent current/flux normalization.
- Coil feasibility and fast-particle confinement remain ESSOS or external
  calculations.

Nonconverged solvers return inspectable result/report objects. Production
callers must reject false `converged` or `finite` flags and must enforce their
own conditioning and resolution thresholds.
