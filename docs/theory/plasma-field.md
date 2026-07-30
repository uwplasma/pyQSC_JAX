# Matched free-space plasma field

The on-axis plasma field is not determined by local Ampère-law data alone.
Remote parts of the current distribution add a harmonic field. The decay
condition at spatial infinity selects that field uniquely and leads to a
matched volume-current Biot–Savart calculation.

`regularized_axis_integral` evaluates the full-torus periodic finite part

\[
\mathcal R(\varphi)=
\int_0^{2\pi}
\left[
L\frac{\mathbf t(\varphi')\times
[\mathbf r_0(\varphi)-\mathbf r_0(\varphi')]}
{|\mathbf r_0(\varphi)-\mathbf r_0(\varphi')|^3}
-\frac{\kappa(\varphi)\mathbf b(\varphi)}
{4|\sin[(\varphi'-\varphi)/2]|}
\right]d\varphi'.
\]

All field-period copies are included. At the coincident grid point the
symmetric finite-part value is used; the two one-sided limits are equal and
opposite.

The inner elliptical channel supplies the finite constants

\[
C_b=-\frac12-\frac12\log\frac{\mathcal T+2}{4}
+\frac{x^2+1}{\mathcal T+2},
\qquad
C_n=-\frac{\chi\sigma}{\mathcal T+2},
\]

and a smooth angular correction \(\mathbf S\) containing the first-harmonic
weighted current and the second-order cross-sectional displacement. The
assembled leading field is

\[
\mathbf B_p(\mathbf r_0)=
\frac{ja^2}{4}
\left[
\mathcal R+\kappa\mathbf b
\left(\log\frac{8L}{a}+C_b\right)
+\kappa C_n\mathbf n
\right]
+\mathbf S
+O(a^4\log a).
\]

`matched_plasma_field_kernel` also exposes the same result at an arbitrary
matching reference length. Changing that length shifts the finite part and
the local logarithm separately but leaves their sum invariant.

Validation includes:

- exact cancellation of the arbitrary matching length;
- a circular-axis finite part below \(10^{-13}\);
- vanishing circular-ellipse core constants;
- the circular local-induction logarithm;
- vacuum and pressure-only limits;
- angular and toroidal resolution convergence;
- JIT and radius JVP checks; and
- an independent resolved volume-current Biot–Savart integral whose error
  follows the predicted \(a^4|\log a|\) scaling.

The returned `estimated_field_remainder` is an order-of-magnitude asymptotic
scale, not a rigorous error bound. `formal_radius_to_curvature_radius` should
be inspected before treating the slender-channel expansion as accurate.

## Local elliptical gradient

At leading gradient order the current channel is locally straight. In the
ordered Frenet basis \((\mathbf t,\mathbf n,\mathbf b)\), with the first tensor
index denoting derivative direction, the manuscript gives

\[
D^p=\frac{j}{\mathcal T+2}
\begin{pmatrix}
0&0&0\\
0&\chi\sigma&1+(1+\sigma^2)/x^2\\
0&-(1+x^2)&-\chi\sigma
\end{pmatrix}.
\]

The canonical package tensor is field-component-first, so
`gradient_frenet` is the transpose of this matrix.
`elliptical_channel_gradient` implements the formula independently of a
near-axis solution and optionally rotates it into Cartesian coordinates.

The local tensor satisfies

\[
\nabla\cdot\mathbf B_p=0,\qquad
D^p_{nb}-D^p_{bn}=j=\mu_0J_\parallel(0).
\]

`plasma_gradient_on_axis` subtracts it from the total near-axis gradient.
The resulting external gradient is symmetric and trace-free within the
spectral resolution of the total solve. The result also supplies its five
independent Cartesian STF components in the order
`(xx, yy, xy, xz, yz)`.

## Curved-channel Hessian

The plasma Hessian cannot be obtained by naively differentiating the singular
Biot–Savart kernel through the current-carrying region. Those derivatives
produce local contact terms. `plasma_hessian_on_axis` instead evaluates the
cubic interior logarithmic potential of the resolved elliptical channel,
including three separately determined contributions:

- the affine first-harmonic weighted current;
- the universal curved-channel metric term; and
- the second-order deformation of the current cross-section.

The transverse derivatives are evaluated algebraically in the oriented
principal axes of the ellipse. Tangential derivatives are obtained by
spectrally differentiating the regular plasma gradient and applying the
Frenet connection. The public tensors use field-component-first ordering,

\[
H_{ijk}=\partial_j\partial_k B_i.
\]

Subtracting the plasma Hessian from the regular-coordinate total Hessian gives
the external target. In a vacuum neighborhood this tensor is fully symmetric
and trace-free:

\[
H^c_{ijk}=H^c_{(ijk)},\qquad H^c_{iik}=0.
\]

`external_hessian_independent` packs its seven Cartesian STF components in the
order `(xxx, xxy, xxz, xyy, xyz, yyy, yyz)`. The corresponding
`pack_symmetric_trace_free_rank3` and
`unpack_symmetric_trace_free_rank3` functions are reversible.

Validation covers the circular finite-conductor curvature limit, QA and QH
oriented ellipses, vacuum reduction, spectral convergence of the Maxwell
identities, JIT/JVP agreement, and all rank-three tensor permutations. The
resolved-volume field test above validates the free-space source and matching
normalization. It is intentionally not replaced by differentiation of the
singular quadrature, which would omit the contact terms that the interior
potential supplies.

As for the lower-order jet, `estimated_hessian_remainder` is a conservative
asymptotic scale rather than a rigorous bound. The predicted remainder is
\(O(a^2|\log a|)\) for fixed near-axis inputs.
