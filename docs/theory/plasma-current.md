# Positive-volume plasma-current source

The free-space Biot–Savart calculation must integrate the physical volume
measure. Expanding current density and the polar-coordinate Jacobian
separately makes both expressions look singular at \(r=0\) and makes it easy
to lose a factor of \(r\). The attached derivation instead defines the regular,
positive-volume source

\[
\mathbf W=\chi\mu_0\mathcal J_r\mathbf J,
\qquad \chi=s_Gs_\psi,
\]

and obtains the exact identity

\[
\mathbf W=\chi\left[
(I_r-r\bar B\beta_\vartheta)\mathbf x_\varphi
-(G_r+NI_r-r\bar B\beta_\varphi)\mathbf x_\vartheta
\right].
\]

`plasma_current_source` expands this identity directly:

\[
\frac{\mathbf W}{L}
=r\mathbf w_1+r^2\mathbf w_2+O(r^3),
\qquad
\mathbf w_1=j\mathbf t,
\qquad
j=2\chi I_2=\mu_0J_\parallel(0).
\]

The quadratic term retains the independently pressure-driven
\(\beta_{1s}\) and \(C_2=G_2+NI_2\) paths. It is represented by cosine and
sine vector coefficients without division by \(I_2\), so it remains regular
as the on-axis parallel current passes through zero.

For a mandatory formal current-channel radius \(a>0\),

\[
\mu_0 I_p(a)=\pi a^2j=2\pi\chi I_2a^2.
\]

`enclosed_current_from_covariant` and `covariant_current_from_enclosed`
implement this conversion. Holding \(I_2\) fixed while varying \(a\) holds the
on-axis current density fixed and makes the enclosed current scale as \(a^2\).
Holding enclosed amperes fixed instead requires \(I_2\propto a^{-2}\), which
is a different asymptotic ordering.

`evaluate_weighted_current` evaluates the regular truncated source at any
radial/helical-angle batch without constructing a finite-radius surface.
