# Physics traceability

Every implemented physics block must point to a primary derivation, a code
location, and at least one independent validation. Equation numbers below refer
to the cited source, not to a duplicated derivation in this repository.

| Physics block | Primary source | Reference implementation | Required validation | Status |
| --- | --- | --- | --- | --- |
| General Fourier axis and Frenet frame | Landreman & Sengupta (2018), arXiv:1809.10233 | `Axis`, `evaluate_axis`, `compute_axis_geometry`; pyQSC `init_axis.py` | analytic circle; asymmetric Fortran curvature/torsion/varphi; frame identities; JIT/VMAP/grad | implemented and documented |
| First-order QS and sigma equation | Landreman & Sengupta (2019), arXiv:1908.10253 | `first_order.solve_sigma`; pyQSC `solve_sigma_equation.py` | residual, convergence report, QA/QH/finite-current pyQSC values, JIT/VMAP/JVP/VJP, finite differences | implemented and documented |
| Field gradient and `L_grad_B` | Landreman (2021), arXiv:2012.00865, Eq. 3.12 | `first_order.first_order_solution`; pyQSC `grad_B_tensor.py` | upstream arrays, vacuum symmetry/divergence, resolution convergence | implemented and documented |
| Complete second-order coefficients | Landreman & Sengupta (2019), arXiv:1908.10253 | `second_order.solve_second_order`; pyQSC `calculate_r2.py`; manuscript Eqs. 36–54 | four independent equation residuals; vacuum QA, finite-pressure/current, QH upstream arrays; JIT/JVP/VJP/finite differences | implemented and documented |
| Total field Hessian | manuscript Eqs. 55–83 | pyQSC generated tensor as comparison only | regular-coordinate chain rule, AD/finite differences, symmetry/divergence | not implemented |
| Third-order flux constraint and shear | Landreman & Sengupta (2019); Rodríguez et al. shear equations | pyQSC `calculate_r3.py` | upstream r3 arrays and boundary data | not implemented |
| Singular radius and scale lengths | Landreman (2021), arXiv:2012.00865 | pyQSC `r_singularity.py` | upstream values; actual coordinate-map regularity | not implemented |
| Good-stellarator criteria | Curvo, Ferreira & Jorge (2025), Table 3 | none canonical | exact normalized thresholds and margins | not implemented |
| Plasma current source | manuscript Eqs. 88–97 | none | normalization, enclosed current, straight circular/elliptic channels | not implemented |
| Plasma on-axis field | manuscript Eqs. 136–138 | none | full-torus periodicity, matching cancellation, resolved volume Biot–Savart | not implemented |
| Plasma gradient | manuscript Eqs. 156–160 | none | Ampère antisymmetry, divergence, external STF projection | not implemented |
| Plasma Hessian | manuscript Eqs. 194 and 210–219 | none | resolved volume Biot–Savart, full symmetry, trace-free external tensor | blocked by validation gate |
| External vacuum jet | manuscript Eqs. 220–224 | none | 3+5+7 representation, asymptotic error scaling, vacuum reduction | not implemented |

## Source policy

pyQSC is BSD-2-Clause licensed. Adapted code, formulas expressed as code, and
derived reference data must preserve attribution. The attached manuscript is
not committed. Its locally audited SHA-256 is recorded in a gitignored note.
