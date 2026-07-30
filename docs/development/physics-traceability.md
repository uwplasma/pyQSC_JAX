# Physics traceability

Every implemented physics block must point to a primary derivation, a code
location, and at least one independent validation. Equation numbers below refer
to the cited source, not to a duplicated derivation in this repository.

| Physics block | Primary source | Reference implementation | Required validation | Status |
| --- | --- | --- | --- | --- |
| General Fourier axis and Frenet frame | Landreman & Sengupta (2018), arXiv:1809.10233 | `Axis`, `evaluate_axis`, `compute_axis_geometry`; pyQSC `init_axis.py` | analytic circle; asymmetric Fortran curvature/torsion/varphi; frame identities; JIT/VMAP/grad | implemented and documented |
| First-order QS and sigma equation | Landreman & Sengupta (2019), arXiv:1908.10253 | `first_order.solve_sigma`; pyQSC `solve_sigma_equation.py` | residual, convergence report, QA/QH/finite-current pyQSC values, JIT/VMAP/JVP/VJP, finite differences | implemented and documented |
| Target-transform inverse solve and continuation | implicit-function reformulation of the first-order sigma equation | `inverse.solve_target_iota`; `continuation.continue_etabar_branch` | `etabar` and `I2` round trips, sign branches, distinct local basins, fold crossing, independent forward correction, r2 propagation, JIT/JVP/finite differences | branch-local inverse and fixed-step `etabar` pseudo-arclength implemented |
| Nonconstant \(B_{20}\) and affine \(B_{2c}\) elimination | Landreman & Sengupta (2019), complete r2 system | `optimize.b20_diagnostics`; `optimize.optimize_B2c` | direct norm definitions, affine reconstruction, stationary optimum, circular degeneracy, spectral tail, doubled-grid verification, JIT/JVP/finite differences | implemented and documented |
| Branch-aware axis search | nonnegative \(B_{20}\) residual plus the package's traced r2 equations | `axis_optimization.search_axis`; `axis_optimization.continue_axis_search` | deterministic batched exploration, bounded LM steps, accepted/rejected damping, criteria failure, basin clustering, certified zero, nonzero best-found semantics, staged modes | implemented and documented |
| Field gradient and `L_grad_B` | Landreman (2021), arXiv:2012.00865, Eq. 3.12 | `first_order.first_order_solution`; pyQSC `grad_B_tensor.py` | upstream arrays, vacuum symmetry/divergence, resolution convergence | implemented and documented |
| Complete second-order coefficients | Landreman & Sengupta (2019), arXiv:1908.10253 | `second_order.solve_second_order`; pyQSC `calculate_r2.py`; manuscript Eqs. 36–54 | four independent equation residuals; vacuum QA, finite-pressure/current, QH upstream arrays; JIT/JVP/VJP/finite differences | implemented and documented |
| Total field Hessian | manuscript Eqs. 55–83 | `field.total_field_jet`; pyQSC generated tensor as comparison only | regular-coordinate chain rule, JIT/JVP, QA and finite-current upstream arrays, full vacuum symmetry, divergence and derivative-of-divergence | implemented and documented |
| Magnetic well and Mercier terms | Landreman (2021), arXiv:2012.00865 | `diagnostics.mercier_diagnostics`; pyQSC `mercier.py` | vacuum and finite-pressure/current upstream values | implemented and documented |
| Third-order flux constraint | Landreman & Sengupta (2019) | `third_order.solve_third_order`; pyQSC `calculate_r3.py` | two independent flux-constraint forms; QA, finite-pressure/current, and QH upstream arrays; r3 boundary data; JIT/JVP/finite differences | implemented and documented |
| Magnetic shear | Rodríguez, Sengupta & Bhattacharjee (2022), Appendix F | `shear.solve_magnetic_shear`; pyQSC `calculate_r3.py` | three upstream paper cases, asymmetric/secular integration, resolution convergence, `B31c`, JIT/JVP/VJP and full-solve AD | implemented and documented for standard MHS with positive signs |
| Singular radius and scale lengths | Landreman (2021), arXiv:2012.00865 | `singularity.singularity_diagnostics`; pyQSC `r_singularity.py` as comparison | direct regular-map determinant, QA/QH/finite-current upstream radii and arrays, residual, angular convergence, JIT/JVP/finite differences | implemented and documented |
| Good-stellarator criteria | Curvo, Ferreira & Jorge (2025), Table 3 | `criteria.Criteria.from_curvo_2025` | exact normalized thresholds, strict/inclusive comparisons, signed margins, scaling, JIT/JVP/finite differences | implemented and documented |
| Plasma current source | manuscript Eqs. 88–97 | `plasma.plasma_current_source`; `plasma.evaluate_weighted_current` | exact \(I_2\)/ampere normalization, regular radial power, pressure-only and vacuum limits, batch/JIT evaluation | implemented and documented |
| Plasma on-axis field | manuscript Eqs. 118 and 133–138 | `plasma.regularized_axis_integral`; `plasma.plasma_field_on_axis` | circular finite part/core/logarithm, full-torus construction, matching-length cancellation, angular/toroidal convergence, vacuum/pressure limits, JIT/JVP, resolved volume Biot–Savart \(a^4|\log a|\) scaling | implemented and documented |
| Plasma gradient | manuscript Eqs. 142–160 | `plasma.elliptical_channel_gradient`; `plasma.plasma_gradient_on_axis` | straight circular and sheared elliptical channels, frame covariance, Ampère antisymmetry, divergence, external symmetry/trace, vacuum reduction, 5-component pack/unpack, JIT/JVP/finite differences | implemented and documented |
| Plasma Hessian | manuscript Eqs. 194 and 210–219 | `plasma.plasma_hessian_on_axis` | circular finite-conductor contact-term limit, QA/QH oriented ellipses, commuting plasma derivatives, spectral convergence, JIT/JVP/finite differences | implemented and documented |
| External vacuum jet | manuscript Eqs. 218–224 | nested `PlasmaHessianData`; rank-2/rank-3 STF packers | 3+5+7 representation, full Hessian symmetry, trace-free gradient/Hessian, vacuum reduction, asymptotic error metadata | implemented and documented; normalized ESSOS objective remains |
| VMEC fixed-boundary conversion | near-axis surface expansion plus VMEC Fourier boundary convention | `vmec.uniform_cylindrical_surface`; `vmec.vmec_boundary`; upstream pyQSC `to_vmec.py` as an independent comparison | vectorized/legacy surface agreement, Fourier reconstruction, asymmetric coefficients, deterministic input, warm timing, frozen and live VMEC on-axis iota and force residuals, radius convergence | implemented and documented |

## Source policy

pyQSC is BSD-2-Clause licensed. Adapted code, formulas expressed as code, and
derived reference data must preserve attribution. The attached manuscript is
not committed. Its locally audited SHA-256 is recorded in a gitignored note.
