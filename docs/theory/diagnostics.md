# Diagnostics

The package keeps diagnostics distinct from solver success:

- `RootSolveReport` and `LinearSolveReport` record residuals, iterations,
  conditioning, finiteness, and convergence;
- `GeometryDiagnostics` records speed, curvature, cylindrical radius, and
  frame validity;
- `MercierDiagnostics` separates magnetic-well and geodesic contributions;
- `SingularityDiagnostics` finds the first zero of the quadratic regular-map
  Jacobian and reports its residual;
- `B20Diagnostics` reports weighted \(L^2\), smooth and grid maxima,
  peak-to-peak variation, Fourier coefficients, and tail decay;
- `CriteriaReport` returns every measurement, threshold, comparison, margin,
  and pass flag.

The detailed definitions are in [field tensors](field-jet.md),
[\(B_{20}\) optimization](b20-optimization.md), and
[named criteria](criteria.md). A small residual is meaningful only with a
finite, well-conditioned solve and independent resolution verification.
