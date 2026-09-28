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

## Independent high-resolution rerun

At `nphi=121`, the current implementation was compared directly in one
process with the audited upstream checkout for vacuum QA,
finite-pressure/current, and QH cases. Across all three cases:

- scalar transform and \(B_{20}\) diagnostics agreed within
  \(8.1\times10^{-12}\);
- sigma, curvature, and torsion agreed within \(3.4\times10^{-14}\);
- the largest absolute difference among `X20`, `Y20`, and `B20` arrays was
  \(1.11\times10^{-11}\).

The comparison used upstream commit
`cd75359ea47548d5db7ccb458c100085c04ba1bc`, JAX 64-bit mode, and rebuilt
both implementations at the stated resolution rather than comparing with
interpolated frozen data.
