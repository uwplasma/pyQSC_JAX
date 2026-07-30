# Choosing a model

| Goal | Order/API | Required checks |
| --- | --- | --- |
| Transform, elongation, \(L_{\nabla B}\) | `order="r1"` | root convergence, geometry validity |
| \(B_{20}\), Mercier, Hessian, singular radius | `order="r2"` | root and linear reports, resolution |
| Flux-consistent boundary and magnetic shear | `order="r3"` plus `solve_magnetic_shear` | r3 residuals and applicable sign restrictions |
| Prescribed transform | `solve_for="etabar"` or `"I2"` | local response and fold flag |
| Surface-free plasma/external target | `plasma_hessian_on_axis` | positive formal radius and asymptotic metadata |
| Legacy ESSOS consumer | `near_axis` adapter | compatibility orientation and mutable-DOF contract |

Use the lowest order that supplies the observable being optimized. Higher
order adds physical information but also tighter geometry, conditioning, and
resolution requirements. A near-axis model is asymptotic; it does not by
itself prove finite-radius equilibrium, coil feasibility, or particle
confinement.
