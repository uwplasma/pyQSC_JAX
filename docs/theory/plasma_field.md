# Plasma field and external target

The near-axis solution gives the **total** field jet on the axis. Coil design needs the
part produced by the coils. This page computes the free-space field of the plasma current,
its gradient and its Hessian on the axis, and subtracts them from the total jet; what
remains is the external vacuum target, $3 + 5 + 7$ independent numbers per axis point.
Nothing requires a finite-radius surface, but a **formal radius** $a > 0$ of the current
channel is required: it fixes the enclosed current and the matching scale.

Notation: $\ell' = |G_0|/B_0$, $\chi = s_Gs_\psi$, $x = X_{1c}$, $y = Y_{1s}$,
$y_\sigma = Y_{1c} = y\sigma$, a subscript $s$ is $d/d\ell = \ell'^{-1}d/d\varphi$, and

$$
\mathcal T = x^2 + \frac{1 + \sigma^2}{x^2}
$$

is the trace of the normalized ellipse matrix.

## Current source

The Biot-Savart volume integral needs the current times the volume Jacobian. Expanding
them separately makes both look singular at $r = 0$. Instead
{func}`~pyqsc_jax.plasma_current_source` expands the regular weighted current
$\mathbf W = \chi\mu_0\mathcal J_r\mathbf J$:

$$
\frac{\mathbf W}{\ell'} = r\,\mathbf w_1 + r^2\left(\mathbf w_{2c}\cos\vartheta + \mathbf w_{2s}\sin\vartheta\right) + O(r^3),
\qquad \mathbf w_1 = j\,\mathbf t,\qquad j = 2\chi I_2 = \mu_0J_\parallel(0).
$$

With $C_2 = G_2 + N_hI_2$ ($N_h = -Nn_\mathrm{fp}$), the quadratic coefficients in the
$(\mathbf t,\mathbf n,\mathbf b)$ frame are

$$
\begin{aligned}
\mathbf w_{2c} &= \left(-\chi s_\psi B_0\beta_{1s} - j\kappa x\right)\mathbf t
 + j\left(x_s - \tau y_\sigma\right)\mathbf n
 + \left[j\left(y_{\sigma,s} + \tau x\right) - \frac{2\chi C_2y}{\ell'}\right]\mathbf b,\\
\mathbf w_{2s} &= \left(-j\tau y + \frac{2\chi C_2x}{\ell'}\right)\mathbf n
 + \left(j\,y_s + \frac{2\chi C_2y_\sigma}{\ell'}\right)\mathbf b .
\end{aligned}
$$

They contain no division by $I_2$, so the pressure-driven part survives at $I_2 = 0$
($\mathbf w^\star_{2c}$ denotes $\mathbf w_{2c}$ without the $-j\kappa x\,\mathbf t$ term).
The enclosed toroidal current is

$$
\mu_0 I_p(a) = \pi a^2 j = 2\pi\chi I_2a^2 .
$$

Holding $I_2$ fixed while varying $a$ holds the on-axis current density fixed; holding
amperes fixed needs $I_2\propto a^{-2}$, a different ordering.

## Matched field on the axis

Local data alone do not fix the plasma field: distant current adds a harmonic field, and
decay at infinity selects it. Matching the Biot-Savart integral of the whole torus to the
field of a slender elliptical channel gives

$$
\mathbf B_p(\mathbf r_0) = \frac{ja^2}{4}\left[\mathcal R + \kappa\mathbf b\left(\log\frac{8\ell'}{a} + C_b\right)
 + \kappa C_n\mathbf n\right] + \mathbf S + O(a^4\log a),
$$

$$
C_b = -\frac12 - \frac12\log\frac{\mathcal T + 2}{4} + \frac{x^2 + 1}{\mathcal T + 2},\qquad
C_n = -\frac{\chi\sigma}{\mathcal T + 2}.
$$

**Regularized axis integral.** $\mathcal R$ is the periodic finite part over the full torus
(all field periods),

$$
\mathcal R(\varphi) = \int_0^{2\pi}\left[\ell'\,\frac{\mathbf t(\varphi')\times\left(\mathbf r_0(\varphi) - \mathbf r_0(\varphi')\right)}{|\mathbf r_0(\varphi) - \mathbf r_0(\varphi')|^3}
 - \frac{\kappa(\varphi)\mathbf b(\varphi)}{4|\sin((\varphi' - \varphi)/2)|}\right]d\varphi' .
$$

{func}`~pyqsc_jax.regularized_axis_integral` evaluates it as follows.

1. The angle is the spectral antiderivative of $d\varphi/d\phi$ (exact increment
   $2\pi/n_\mathrm{fp}$ per period), not the trapezoidal `geometry.varphi`, whose
   $O(h^2)$ error would set a floor on the integral.
2. The bounded integrand behaves near the coincident node as
   $c\,\mathrm{sign}(s) + a_1|s| + \text{smooth} + O(|s|^3)$. The coincident node is given
   the value 0, the mean of the two one-sided limits
   $\pm\ell'(\kappa_s\mathbf b - \kappa\tau\mathbf n)/3$, so the jump cancels in the symmetric
   sum.
3. The kink $a_1|s|$ gives the punctured trapezoidal rule the error
   $2\zeta(-1)a_1h^2 = -a_1h^2/6$ (generalized Euler-Maclaurin
   {cite}`sidi1988quadrature`). With $w_k$ the weighted samples $k$ nodes from the
   observation point, the correction
   $\left[4(w_{1} + w_{-1}) - (w_{2} + w_{-2})\right]/24$ equals $a_1h^2/6$ and vanishes for
   the smooth $h^2$ term, leaving $O(h^4)$.

The convergence figure on {doc}`second_order` (panel c) shows both rates; at $n_\phi = 61$
the error is {{ convergence_integral_error_61 }} with the correction and
{{ convergence_integral_plain_61 }} without.

**Matching length.** The code evaluates the bracket at an arbitrary matching length
$L_\mathrm{ref}$ as
$\mathcal R - \kappa\mathbf b\log\frac{L_\mathrm{ref}}{4\ell'} + \kappa\mathbf b\left(\log\frac{2L_\mathrm{ref}}{a} + C_b\right) + \kappa C_n\mathbf n$;
the two logarithms cancel in $L_\mathrm{ref}$. `maximum_matching_scale_error` compares
$L_\mathrm{ref} = 4\ell'$ and $7\ell'$: {{ db_matching_scale_error }} for database
configuration 52521.

**Shape correction.** With
$\mathbf e = x\cos\vartheta\,\mathbf n + (y\sin\vartheta + y_\sigma\cos\vartheta)\mathbf b$,
$\boldsymbol\xi_2 = X_2\mathbf n + Y_2\mathbf b + Z_2\mathbf t$ and
$\mathbf w^\star_2 = \mathbf w^\star_{2c}\cos\vartheta + \mathbf w_{2s}\sin\vartheta$,

$$
\mathbf S = -\frac{a^2}{2}\left\langle\frac{\mathbf w_1\times\boldsymbol\xi_2 + \mathbf w^\star_2\times\mathbf e}{|\mathbf e|^2}
 - \frac{2(\mathbf e\cdot\boldsymbol\xi_2)\,\mathbf w_1\times\mathbf e}{|\mathbf e|^4}\right\rangle_\vartheta,
$$

averaged over `angular_resolution` (default 128) angles. It carries the pressure-driven
field when $I_2 = 0$.

`estimated_field_remainder` $=\max|\mathbf B_p|\,(a/\ell')^2(1 + |\log(a/\ell')|)$ is an
order-of-magnitude indicator of the omitted relative order, not a bound;
`formal_radius_to_curvature_radius` $= a\max\kappa$ should be small.

## Gradient

**Finite current.** To leading order the channel is locally straight and elliptical. In the
$(\mathbf t,\mathbf n,\mathbf b)$ basis, with the first index the derivative direction,

$$
D^p = \frac{j}{\mathcal T + 2}\begin{pmatrix}0 & 0 & 0\\ 0 & \chi\sigma & 1 + (1 + \sigma^2)/x^2\\ 0 & -(1 + x^2) & -\chi\sigma\end{pmatrix},
$$

which satisfies $\nabla\cdot\mathbf B_p = 0$ and $D^p_{nb} - D^p_{bn} = j = \mu_0J_\parallel(0)$
(Ampère). The package stores the field-first transpose. For a finite-current case the
Ampère residual is {{ current_ampere_error }}.

**Zero current, order $a^2$.** At $I_2 = 0$ the leading gradient vanishes, and the first
nonzero gradient is $O(a^2)$, driven by the pressure current. With
$\hat p = \mu_0p_2/B_0$, $w = 1 + x^2 + i\chi\sigma$ and complex amplitudes

$$
f_t = \frac{4s_\psi\hat p\,\ell'\bar\eta\,x}{\iota_Nw},\qquad
f_n = \frac{2is_G\hat p\,x^2}{w},\qquad
f_b = \frac{2s_G\hat p(1 + i\chi\sigma)}{w},
$$

$$
\begin{aligned}
\Sigma &= X_{2c} + \chi Y_{2s} + i(Y_{2c} - \chi X_{2s}),\\
q_t &= \frac{x^2}{2w^2}\left[s_\psi\hat p\left(8Z_{2s} - \frac{\ell'}{\iota_N}\left(10\bar\eta^2 - \frac{8B_{2c}}{B_0}\right)\right) + 8is_G\hat pZ_{2c} - 4f_t\Sigma\right],
\end{aligned}
$$

and $c = \kappa x^2f_t/(4w)$, the transverse components are

$$
\begin{aligned}
g_{nn} &= a^2\left[-\operatorname{Im}\left(q_t + \tfrac14\kappa f_t + c\right) - \tfrac12\operatorname{Re}f_{b,s}
 - \tfrac12\tau\left(\operatorname{Re}f_n + \operatorname{Im}f_b\right)\right],\\
g_{nb} &= a^2\left[\tfrac12\operatorname{Re}f_{n,s} - \tfrac12\tau\operatorname{Re}f_b + \tfrac12\tau\operatorname{Im}f_n
 - \operatorname{Re}(q_t + c)\right],
\end{aligned}
$$

the tangential row is the derivative of the matched field along the axis,
$g_{tt} = B_{t,s} - \kappa B_n$, $g_{tn} = B_{n,s} + \kappa B_t - \tau B_b$,
$g_{tb} = B_{b,s} + \tau B_n$, and $g_{bb} = -g_{tt} - g_{nn}$; the tensor is symmetric
and trace-free. It is valid for stellarator-symmetric solutions with $I_2 = 0$.

{func}`~pyqsc_jax.plasma_gradient_on_axis` selects (does not sum) the two branches with
`I2 == 0`. The retained order changes there, so the gradient jumps as $I_2\to 0$: for
configuration 52521 at $a =$ {{ db_formal_radius }} m the jump is {{ zero_current_jump }} T/m,
the full size of the order-$a^2$ gradient. This is a truncation artefact. Hold $I_2$
fixed when optimizing and do not differentiate with respect to $I_2$ near zero.

**External gradient.** $\nabla\mathbf B_c = \nabla\mathbf B_\mathrm{tot} - \nabla\mathbf B_p$
should be symmetric and trace-free. For 52521 its asymmetry is
{{ db_external_gradient_asymmetry }} and its trace {{ db_external_gradient_trace }} T/m. It is
projected to STF form and packed as five numbers $(xx, yy, xy, xz, yz)$.

## Hessian

Differentiating the singular Biot-Savart kernel through the current region would miss
local contact terms. {func}`~pyqsc_jax.plasma_hessian_on_axis` instead builds the cubic
interior vector potential $(A_t, A_n, A_b)$ of the elliptical channel in the physical normal
and binormal distances $(u, v)$ and differentiates it twice. The cubic potential of an
affine current density $\alpha_nu + \alpha_bv$ has the coefficients $\Phi(\alpha_n,\alpha_b)$ of
$(u^3, u^2v, uv^2, v^3)$

$$
\begin{aligned}
\Phi_{u^3} &= \tfrac{\pi}{12}\left[(3 - 2h_c - h_c^2 + h_s^2)\alpha_n - 2h_s(1 - h_c)\alpha_b\right],\\
\Phi_{u^2v} &= \tfrac{\pi}{4}\left[2h_s(1 + h_c)\alpha_n + (1 - 2h_c + h_c^2 - h_s^2)\alpha_b\right],\\
\Phi_{uv^2} &= \tfrac{\pi}{4}\left[(1 + 2h_c + h_c^2 - h_s^2)\alpha_n + 2h_s(1 - h_c)\alpha_b\right],\\
\Phi_{v^3} &= \tfrac{\pi}{12}\left[-2h_s(1 + h_c)\alpha_n + (3 + 2h_c - h_c^2 + h_s^2)\alpha_b\right],
\end{aligned}
$$

fixed by Poisson's equation and by matching to the decaying exterior potential. Here
$h_c = (x^4 - 1 - \sigma^2)/\Delta$ and $h_s = -2\chi\sigma x^2/\Delta$, with
$\Delta = (1 + x^2)^2 + \sigma^2$, describe the anisotropy of the ellipse, so nothing is
diagonalized and the result stays differentiable through a circular section. Three sources
contribute:

1. the affine weighted current, with $\alpha_n = (c - \sigma s)/x$, $\alpha_b = \chi xs$ for
   each Frenet component ($c$, $s$) of $\mathbf w^\star_{2c}$, $\mathbf w_{2s}$:
   $\mathbf A^{(1)} = -\Phi/(2\pi)$;
2. curvature: $A_t \mathrel{+}= \frac{j\kappa}{4\pi}\left[\Phi(1, 0) - \pi(Q_1, Q_2, Q_3, 0)\right]$ with
   $Q = (1 + x^2 + \sigma^2,\ -2\chi\sigma x^2,\ x^2(1 + x^2))/\Delta$;
3. the second-order deformation of the section:
   $A_t \mathrel{+}= \frac{j}{2\pi}\left[\Phi(\zeta_n, \zeta_b) - (b_c, 3b_s, -3b_c, -b_s)\right]$ with

   $$
   \begin{aligned}
   \zeta_n &= \tfrac{2}{x^2}\left[(1 + \sigma^2)X_{20} + (1 - \sigma^2)X_{2c} - 2\sigma X_{2s}\right] + 2\chi(Y_{2s} - \sigma Y_{20} + \sigma Y_{2c}),\\
   \zeta_b &= 2\chi(X_{2s} - \sigma X_{20} + \sigma X_{2c}) + 2x^2(Y_{20} - Y_{2c}),
   \end{aligned}
   $$

   $F_1 = X_{2c} + \sigma X_{2s} - \chi x^2Y_{2s}$, $F_2 = \chi X_{2s} - \chi\sigma X_{2c} + x^2Y_{2c}$,
   $c_3 = (1 + x^2)^3 - 3(1 + x^2)\sigma^2$, $s_3 = 3(1 + x^2)^2\sigma - \sigma^3$,
   $\beta = -4\pi x^2/(3\Delta^3)$, $b_c = \beta(F_1c_3 - \chi F_2s_3)$, $b_s = \beta(\chi F_1s_3 + F_2c_3)$.

With the uniform-current quadratic potential $m = -\frac j2Q$ and
$B_t = \partial_uA_b - \partial_vA_n$, $B_n = \partial_vA_t$, $B_b = \kappa A_t - \partial_uA_t$,
the transverse second derivatives are (index $k$ of $a_{\cdot k}$ is the monomial
$u^3, u^2v, uv^2, v^3$)

| | $B_t$ | $B_n$ | $B_b$ |
|---|---|---|---|
| $\partial_{nn}$ | $6a_{b0} - 2a_{n1}$ | $2a_{t1}$ | $2\kappa m_0 - 6a_{t0}$ |
| $\partial_{nb}$ | $2a_{b1} - 2a_{n2}$ | $2a_{t2}$ | $\kappa m_1 - 2a_{t1}$ |
| $\partial_{bb}$ | $2a_{b2} - 6a_{n3}$ | $6a_{t3}$ | $2\kappa m_2 - 2a_{t2}$ |

The rows with a tangential derivative come from the leading gradient $G$ (derivative-first)
with the Frenet connection
$\Omega = \begin{pmatrix}0&\kappa&0\\-\kappa&0&\tau\\0&-\tau&0\end{pmatrix}$:
$\partial_tG = \ell'^{-1}D_\varphi G - \Omega G - G\Omega^T$. The leading gradient is used
even when $I_2 = 0$, so a single tensor never mixes asymptotic orders. The result is
field-component first, $H_{ijk} = \partial_j\partial_kB_i$.

**External Hessian.** $H^c = H^\mathrm{tot} - H^p$ should be fully symmetric and trace-free,
$H^c_{ijk} = H^c_{(ijk)}$, $H^c_{iik} = 0$. For 52521 the symmetry error is
{{ db_external_hessian_symmetry }} and the trace {{ db_external_hessian_trace }} T/m², both
set by the resolution of the total Hessian. It is packed as seven numbers
$(xxx, xxy, xxz, xyy, xyz, yyy, yyz)$; `pack_symmetric_trace_free_rank3` and
`unpack_symmetric_trace_free_rank3` in `pyqsc_jax.plasma` are inverse to each other.
`estimated_hessian_remainder` $=\max|H^p|(a/\ell')^2(1 + |\log(a/\ell')|)$, here
{{ db_hessian_remainder }} T/m², is again an indicator, not a bound.

The heavy independent references (resolved-volume Biot-Savart integrals whose error follows
$a^4|\log a|$) live in the study repository
[plasma-coil-fields](https://github.com/rogeriojorge/plasma-coil-fields/tree/pyqsc-jax-validation/pyqsc_jax_validation). The package tests
check the analytic limits, Maxwell identities, matching-length invariance and derivatives
({doc}`../development/index`).
