# Total on-axis field jet

## Regular coordinates

The total magnetic field, gradient, and Hessian are computed without
constructing a finite-radius surface. Define the regular transverse
coordinates

\[
q_1=r\cos\vartheta,\qquad q_2=r\sin\vartheta.
\]

Through quadratic order the position is

\[
\boldsymbol{x}
=\boldsymbol{r}_0+\boldsymbol{d}_1q_1+\boldsymbol{d}_2q_2
+\frac12\boldsymbol{h}_{11}q_1^2
+\boldsymbol{h}_{12}q_1q_2
+\frac12\boldsymbol{h}_{22}q_2^2.
\]

The vectors \(\boldsymbol d_a\) come from the first-order coefficients and
\(\boldsymbol h_{ab}\) from the complete second-order solution. Consequently,
the coordinate map and its inverse are nonsingular on axis even though polar
coordinates \((r,\vartheta)\) are not.

The implementation follows equations 55–83 of the audited surface-free
plasma/coil derivation. It expands

\[
\boldsymbol B_{\mathrm{tot}}
=P\left[
\boldsymbol x_{,\varphi}
+\iota_N(-q_2\boldsymbol x_{,q_1}+q_1\boldsymbol x_{,q_2})
\right]
\]

through second order in \(q_1,q_2\), then applies the ordinary inverse-map
chain rule. This avoids the long generated component expressions in pyQSC;
those expressions are retained only as an independent regression reference.

## A subtle periodicity rule

Scalar coefficients and cylindrical components are periodic over one field
period. Fixed Cartesian components of a vector generally are not: the
cylindrical basis rotates by \(2\pi/n_{\mathrm{fp}}\). A cylindrical vector
\(\boldsymbol v=(v_R,v_\phi,v_Z)\) is therefore differentiated using

\[
\frac{\mathrm d\boldsymbol v}{\mathrm d\varphi}
=\left(v_R'-\phi'v_\phi\right)\boldsymbol e_R
+\left(v_\phi'+\phi'v_R\right)\boldsymbol e_\phi
+v_Z'\boldsymbol e_Z.
\]

Applying a periodic Fourier derivative directly to Cartesian frame samples
is incorrect for \(n_{\mathrm{fp}}>1\). The connection terms above are part of
the implementation and have dedicated Maxwell-identity and pyQSC regression
tests.

## Array ordering and diagnostics

Canonical arrays place the sample axis first:

- `field[n, i]` is \(B_i\);
- `gradient[n, i, j]` is \(\partial B_i/\partial x_j\);
- `hessian[n, i, j, k]` is
  \(\partial^2B_i/(\partial x_j\partial x_k)\).

`hessian_frenet` follows pyQSC's historical
`(sample, derivative, derivative, field)` ordering in the
`(normal, binormal, tangent)` frame. The compatibility adapter exposes this
array as `grad_grad_B`.

`FieldJet` includes the regular-coordinate Jacobian, inverse Jacobian, and
coordinate Hessian together with:

- field and gradient agreement with the lower-order construction;
- \(\nabla\cdot\boldsymbol B\);
- symmetry in the two derivative indices;
- the gradient of \(\nabla\cdot\boldsymbol B\);
- \(L_{\nabla\nabla B}\) and its inverse.

Vacuum tests additionally require full index symmetry. Finite-current tests
retain only the mixed-spatial-derivative symmetry that Maxwell's equations
require.

## Mercier quantities

`MercierDiagnostics` provides the leading near-axis quantities
`d2_volume_d_psi2`, `DGeod_times_r2`, `DWell_times_r2`, and
`DMerc_times_r2`. Their normalization and signs match pyQSC at the audited
upstream commit. The geodesic and well contributions are kept separate so a
caller can inspect cancellations rather than accepting only their sum.
