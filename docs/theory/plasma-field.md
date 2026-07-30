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
