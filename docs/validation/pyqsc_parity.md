# pyQSC parity

Reference data were generated from landreman/pyQSC commit
`cd753596fd64babfb3832d2a676ca5b80b324b66` and stored locally with
license attribution and checksums. Tests never download regression data.

The parity suite covers:

- axis position, \(\varphi\), curvature, torsion, and helicity;
- sigma, transform, elongation, field and gradient;
- complete vacuum QA, finite-pressure/current, and QH r2 coefficients;
- \(B_{20}\), Mercier, Hessian, and singular radius;
- r3 boundary coefficients and flux-constraint checks;
- magnetic shear reference cases;
- historical adapter shapes, properties, `x`, `dofs`, and boundary behavior.

pyQSC arrays are converted once at the compatibility boundary; canonical
pyQSC_JAX arrays keep the sample axis first. The upstream audit and exact
tolerances are implemented in `tests/regression`, `tests/physics`, and
`tests/compatibility`.

The upstream project is available at
[landreman/pyQSC](https://github.com/landreman/pyQSC).
