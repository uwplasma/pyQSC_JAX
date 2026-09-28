# Field jet and diagnostics

All diagnostics on this page need a second-order solution. They are computed on first
access to the corresponding attribute, or stored by `solve(..., diagnostics=True)`
({doc}`../user_guide/diagnostics`).

## Regular coordinates

The polar coordinates $(r,\vartheta)$ are singular on the axis; the Cartesian-like
$q_1 = r\cos\vartheta$, $q_2 = r\sin\vartheta$ are not. Through second order,

$$
\mathbf x(\varphi, q_1, q_2) = \mathbf r_0 + \mathbf d_1 q_1 + \mathbf d_2 q_2
 + \tfrac12\mathbf h_{11}q_1^2 + \mathbf h_{12}q_1q_2 + \tfrac12\mathbf h_{22}q_2^2,
$$

$$
\begin{aligned}
\mathbf d_1 &= X_{1c}\mathbf n + Y_{1c}\mathbf b,\qquad \mathbf d_2 = Y_{1s}\mathbf b,\\
\mathbf h_{11} &= 2\left[(X_{20} + X_{2c})\mathbf n + (Y_{20} + Y_{2c})\mathbf b + (Z_{20} + Z_{2c})\mathbf t\right],\\
\mathbf h_{12} &= 2\left[X_{2s}\mathbf n + Y_{2s}\mathbf b + Z_{2s}\mathbf t\right],\\
\mathbf h_{22} &= 2\left[(X_{20} - X_{2c})\mathbf n + (Y_{20} - Y_{2c})\mathbf b + (Z_{20} - Z_{2c})\mathbf t\right].
\end{aligned}
$$

## The field in regular coordinates

In Boozer coordinates $\mathbf B = P\left[\mathbf x_\varphi + \iota_N(-q_2\mathbf x_{q_1} + q_1\mathbf x_{q_2})\right]$
with $P = B^2/(G + \iota I)$. Expanding both factors,

$$
\mathbf V = \mathbf V_0 + \mathbf V_1q_1 + \mathbf V_2q_2 + \tfrac12\mathbf V_{11}q_1^2 + \mathbf V_{12}q_1q_2 + \tfrac12\mathbf V_{22}q_2^2,
$$

$$
\begin{aligned}
\mathbf V_0 &= \ell'\mathbf t, &\mathbf V_1 &= \mathbf d_1' + \iota_N\mathbf d_2, &\mathbf V_2 &= \mathbf d_2' - \iota_N\mathbf d_1,\\
\mathbf V_{11} &= \mathbf h_{11}' + 2\iota_N\mathbf h_{12}, &\mathbf V_{12} &= \mathbf h_{12}' + \iota_N(\mathbf h_{22} - \mathbf h_{11}), &\mathbf V_{22} &= \mathbf h_{22}' - 2\iota_N\mathbf h_{12},
\end{aligned}
$$

$$
P = p_0 + p_1q_1 + C_{11}q_1^2 + C_{22}q_2^2,\qquad
p_0 = \frac{B_0^2}{G_0},\quad p_1 = \frac{2B_0^2\bar\eta}{G_0},
$$

$$
\begin{aligned}
C_{11} &= \frac{B_0^2\bar\eta^2 + 2B_0(B_{20} + B_{2c})}{G_0} - F,\qquad
C_{22} = \frac{2B_0(B_{20} - B_{2c})}{G_0} - F,\\
F &= \frac{B_0^2(G_2 + \iota I_2)}{G_0^2}.
\end{aligned}
$$

The $q_1q_2$ term $4B_0B_{2s}/G_0$ of $P$ is not included, so the Hessian assumes
$B_{2s} = 0$ (always the case for stellarator symmetry).

**Periodicity.** Cylindrical components are periodic over a field period; fixed Cartesian
components are not, because the cylindrical basis rotates. A vector
$\mathbf v = (v_R, v_\phi, v_Z)$ is therefore differentiated as

$$
\frac{d\mathbf v}{d\varphi} = \left(v_R' - \frac{d\phi}{d\varphi}v_\phi\right)\mathbf e_R
 + \left(v_\phi' + \frac{d\phi}{d\varphi}v_R\right)\mathbf e_\phi + v_Z'\mathbf e_Z,
$$

with spectral derivatives of the periodic components. Applying $D_\varphi$ directly to
Cartesian samples would be wrong for $n_\mathrm{fp} > 1$.

## Chain rule

With $u = (\varphi, q_1, q_2)$, the coordinate Jacobian on the axis is
$J = [\ell'\mathbf t,\ \mathbf d_1,\ \mathbf d_2]$ and the coordinate Hessian
$\partial^2\mathbf x/\partial u_a\partial u_b$ has entries $\ell'^2\kappa\,\mathbf n$,
$\mathbf d_1'$, $\mathbf d_2'$, $\mathbf h_{11}$, $\mathbf h_{12}$, $\mathbf h_{22}$. Then

$$
\partial_j B_i = \frac{\partial B_i}{\partial u_a}J^{-1}_{aj},\qquad
\partial_j\partial_k B_i = \frac{\partial^2 B_i}{\partial u_a\partial u_b}J^{-1}_{aj}J^{-1}_{bk}
 - \frac{\partial B_i}{\partial u_a}J^{-1}_{al}\frac{\partial^2 x_l}{\partial u_c\partial u_b}J^{-1}_{cj}J^{-1}_{bk},
$$

where $\partial\mathbf B/\partial u$ and $\partial^2\mathbf B/\partial u^2$ come from
$\mathbf B = P\mathbf V$. This replaces the long generated component expressions of pyQSC
with the same result.

{func}`~pyqsc_jax.total_field_jet` returns a {class}`~pyqsc_jax.FieldJet` with `field`,
`gradient`, `hessian` (field component first), `hessian_frenet` (pyQSC order), the
coordinate Jacobian, its inverse and Hessian, and the checks: agreement of field and
gradient with the first-order construction, $\max|\nabla\cdot\mathbf B|$, the asymmetry in
the two derivative indices, $\max|\nabla(\nabla\cdot\mathbf B)|$, and

$$
L_{\nabla\nabla B} = \left(\frac{\|\nabla\nabla\mathbf B\|_F}{4B_0}\right)^{-1/2}
$$

{cite}`landreman2021figures`. For section 5.1 at $n_\phi = 61$: gradient agreement
{{ ls51_jet_gradient_error }}, divergence {{ ls51_jet_divergence }}, index asymmetry
{{ ls51_jet_asymmetry }}, gradient of divergence {{ ls51_jet_divergence_gradient }} and
$\min L_{\nabla\nabla B} =$ {{ ls51_L_grad_grad_B }} m.

## Singular radius

The truncated map $\mathbf x(\varphi,q_1,q_2)$ stops being invertible where its Jacobian
determinant vanishes. Collecting powers of $q_1, q_2$ in
$\det[\mathbf x_\varphi, \mathbf x_{q_1}, \mathbf x_{q_2}]$ by multilinearity gives, through
$O(r^2)$,

$$
g(r,\vartheta) = g_0 + r\left(g_{1c}\cos\vartheta + g_{1s}\sin\vartheta\right)
 + r^2\left(g_{20} + g_{2s}\sin2\vartheta + g_{2c}\cos2\vartheta\right),
$$

each coefficient a sum of triple products of $\ell'\mathbf t$, $\mathbf d_a$,
$\mathbf h_{ab}$ and their $\varphi$ derivatives. At each toroidal sample
{func}`~pyqsc_jax.singularity_diagnostics`:

1. scans 256 angles and takes the smallest positive root in $r$ of the quadratic;
2. uses it (under `stop_gradient`) to seed 8 Newton iterations on
   $g = 0,\ \partial_\vartheta g = 0$ for $(r,\vartheta)$, rejecting non-positive radii;
3. reports $r$, $\vartheta$ and the residual; `r_singularity` is the minimum over $\varphi$.

Derivatives of $r_\mathrm{sing}$ come only from the refined equations, which is why they
are finite: for section 5.1, $r_\mathrm{sing} =$ {{ ls51_r_singularity }} m with residual
{{ ls51_singularity_residual }}, and its derivative with respect to $\bar\eta$ matches a
centred difference to {{ ad_r_singularity_relative_error }}. The singular radius
{cite}`landreman2021figures` bounds where the truncated expansion is a valid coordinate
system. It does not prove that nested surfaces exist up to that radius.

## Mercier terms

With $\mathcal I = \frac{2\pi}{L}\oint\frac{d\ell}{d\phi}
\frac{\bar\eta^4 + \kappa^4\sigma^2 + \bar\eta^2\kappa^2}
{\bar\eta^4 + \kappa^4(1+\sigma^2) + 2\bar\eta^2\kappa^2}d\phi$
over the full torus {cite}`landreman2020magnetic`,

$$
\begin{aligned}
D_\mathrm{Geod}r^2 &= -\frac{2\mu_0^2p_2^2G_0^4\bar\eta^2}{\pi^3B_0^{10}\iota_N^2}\,\mathcal I,\\
\frac{d^2V}{d\psi^2} &= \frac{4\pi^2|G_0|}{B_0^3}\left(3\bar\eta^2 - \frac{4\langle B_{20}\rangle}{B_0}
 + \frac{2(G_2 + \iota I_2)}{G_0}\right),\\
D_\mathrm{Well}r^2 &= \frac{\mu_0p_2|G_0|}{8\pi^4B_0^3}\left(\frac{d^2V}{d\psi^2}
 - \frac{8\pi^2\mu_0p_2|G_0|}{B_0^5}\right),\qquad
D_\mathrm{Merc}r^2 = D_\mathrm{Well}r^2 + D_\mathrm{Geod}r^2 .
\end{aligned}
$$

A positive $D_\mathrm{Merc}$ is Mercier stable; $d^2V/d\psi^2 < 0$ is a magnetic well. The
two contributions are kept separate so cancellations are visible. Normalization and signs
match pyQSC. For database configuration 52521: $D_\mathrm{Well}r^2 =$ {{ db_DWell }} and
$D_\mathrm{Merc}r^2 =$ {{ db_DMerc }}.

## $B_{20}$ diagnostics and exact $B_{2c}$

With arclength weights $w_j = (d\ell/d\phi)_j$ and $\tilde B_{20} = B_{20} - \langle B_{20}\rangle_w$,
{func}`~pyqsc_jax.b20_diagnostics` returns

$$
\frac{1}{B_0}\langle\tilde B_{20}^2\rangle_w^{1/2},\qquad
\frac{1}{B_0}\langle|\tilde B_{20}|^{p}\rangle_w^{1/p}\ (p = 16),\qquad
\frac{\max_j|\tilde B_{20}|}{B_0},\qquad \frac{\max B_{20} - \min B_{20}}{B_0},
$$

and the toroidal Fourier coefficients
$\hat b_m = \langle\tilde B_{20}e^{-imn_\mathrm{fp}\varphi}\rangle_w$, $m = 1,\dots,(n_\phi-1)/2$,
computed by weighted quadrature in the Boozer angle, with their $\ell^2$ and $\ell^1$
norms and the fraction of energy in the top quarter of modes (a resolution check).

For fixed axis and other inputs, the whole second-order solve is affine in $B_{2c}$:
$B_{20} = u + B_{2c}v$. {func}`~pyqsc_jax.optimize_B2c` gets $u$ and $v$ from two solves at
$B_{2c} = 0$ and $1$ and minimizes the weighted variance exactly,

$$
B_{2c}^\star = -\frac{\langle\tilde u\,\tilde v\rangle_w}{\langle\tilde v^2\rangle_w},
$$

keeping the input value when $\langle\tilde v^2\rangle_w \le 10^{-24}$ (e.g. a circular
axis, where $B_{2c}$ only moves the mean). It then recomputes the full solution (and the
$r^3$ correction if present) and reports the affine reconstruction error.
{func}`~pyqsc_jax.optimal_B2c_value` returns only $B_{2c}^\star$, for use inside an
objective.
