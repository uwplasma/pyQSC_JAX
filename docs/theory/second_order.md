# Second order at finite pressure

`order="r2"` solves the complete $O(r^2)$ system of {cite:t}`landreman2019highorder`
(their equations 36-54) with pressure $p_2$ and current $I_2$ both allowed to be nonzero.
The constants $B_{2c}$ and $B_{2s}$ are inputs. Below, $\ell' = |G_0|/B_0$, $s = s_Gs_\psi$,
$c_I = s_\psi I_2/B_0$ and a prime is $d/d\varphi$.

## Explicit stages

1. **Tangential displacement.** With
   $V_1 = X_{1c}^2 + Y_{1c}^2 + Y_{1s}^2$, $V_2 = 2Y_{1s}Y_{1c}$,
   $V_3 = X_{1c}^2 + Y_{1c}^2 - Y_{1s}^2$,

   $$
   Z_{20} = -\frac{V_1'}{8\ell'},\qquad
   Z_{2s} = -\frac{V_2' - 2\iota_N V_3}{8\ell'},\qquad
   Z_{2c} = -\frac{V_3' + 2\iota_N V_2}{8\ell'}.
   $$

2. **Normal second harmonics** from the $B_{2c}$, $B_{2s}$ field-strength equations, with
   $q_s = -\iota_N X_{1c} - \ell'\tau Y_{1s}$, $q_c = X_{1c}' - \ell'\tau Y_{1c}$,
   $r_s = Y_{1s}' - \iota_N Y_{1c}$, $r_c = Y_{1c}' + \iota_N Y_{1s} + \ell'\tau X_{1c}$:

   $$
   \begin{aligned}
   X_{2s} &= \frac{1}{\ell'\kappa}\left[Z_{2s}' - 2\iota_N Z_{2c}
     + \frac{1}{\ell'}\left(\frac{\ell'^2 B_{2s}}{B_0} + \frac{q_cq_s + r_cr_s}{2}\right)\right],\\
   X_{2c} &= \frac{1}{\ell'\kappa}\left[Z_{2c}' + 2\iota_N Z_{2s}
     - \frac{1}{\ell'}\left(-\frac{\ell'^2 B_{2c}}{B_0} + \frac{\ell'^2\bar\eta^2}{2}
     - \frac{q_c^2 - q_s^2 + r_c^2 - r_s^2}{4}\right)\right].
   \end{aligned}
   $$

3. **Pressure response** (the $\sin\vartheta$ part of $\beta$):

   $$
   \beta_{1s} = -\frac{4 s\,\mu_0 p_2\,\bar\eta\,\ell'}{\iota_N B_0^2}.
   $$

## The coupled periodic system

Two area (Jacobian) constraints

$$
\begin{aligned}
0 &= -X_{1c}Y_{2c} + X_{1c}Y_{20} + X_{2s}Y_{1s} + X_{2c}Y_{1c} - X_{20}Y_{1c},\\
0 &= X_{1c}Y_{2s} + X_{2c}Y_{1s} - X_{2s}Y_{1c} + X_{20}Y_{1s} + \tfrac{s}{2}\kappa X_{1c}
\end{aligned}
$$

give $Y_{2s}$ and $Y_{2c}$ as affine functions of $X_{20}$ and $Y_{20}$. The six
force-balance components

$$
\begin{aligned}
f_{X0} &= X_{20}' - \ell'\tau Y_{20} + \ell'\kappa Z_{20} - 4s\ell'(Y_{2c}Z_{2s} - Y_{2s}Z_{2c})\\
 &\quad - c_I\ell'\left(\tfrac12\kappa X_{1c}Y_{1c} - 2Y_{20}\right) + \tfrac12\ell'\beta_{1s}Y_{1c},\\
f_{Xs} &= X_{2s}' - 2\iota_N X_{2c} - \ell'\tau Y_{2s} + \ell'\kappa Z_{2s} - 4s\ell'(Y_{2c}Z_{20} - Y_{20}Z_{2c})\\
 &\quad - c_I\ell'\left(\tfrac12\kappa X_{1c}Y_{1s} - 2Y_{2s}\right) - \tfrac12\ell'\beta_{1s}Y_{1s},\\
f_{Xc} &= X_{2c}' + 2\iota_N X_{2s} - \ell'\tau Y_{2c} + \ell'\kappa Z_{2c} - 4s\ell'(Y_{20}Z_{2s} - Y_{2s}Z_{20})\\
 &\quad - c_I\ell'\left(\tfrac12\kappa X_{1c}Y_{1c} - 2Y_{2c}\right) - \tfrac12\ell'\beta_{1s}Y_{1c},\\
f_{Y0} &= Y_{20}' + \ell'\tau X_{20} - 4s\ell'(X_{2s}Z_{2c} - X_{2c}Z_{2s})\\
 &\quad - c_I\ell'\left(-\tfrac12\kappa X_{1c}^2 + 2X_{20}\right) - \tfrac12\ell'\beta_{1s}X_{1c},\\
f_{Ys} &= Y_{2s}' - 2\iota_N Y_{2c} + \ell'\tau X_{2s} - 4s\ell'(X_{20}Z_{2c} - X_{2c}Z_{20}) - 2c_I\ell' X_{2s},\\
f_{Yc} &= Y_{2c}' + 2\iota_N Y_{2s} + \ell'\tau X_{2c} - 4s\ell'(X_{2s}Z_{20} - X_{20}Z_{2s})\\
 &\quad - c_I\ell'\left(-\tfrac12\kappa X_{1c}^2 + 2X_{2c}\right) + \tfrac12\ell'\beta_{1s}X_{1c}
\end{aligned}
$$

enter the two independent force-balance conditions

$$
\begin{aligned}
0 &= X_{1c}f_{Xs} - Y_{1s}f_{Y0} + Y_{1c}f_{Ys} - Y_{1s}f_{Yc},\\
0 &= -X_{1c}f_{X0} + X_{1c}f_{Xc} - Y_{1c}f_{Y0} + Y_{1s}f_{Ys} + Y_{1c}f_{Yc}.
\end{aligned}
$$

After substituting $Y_{2s}$ and $Y_{2c}$, these are linear in $u = (X_{20}, Y_{20})$ at the
$n_\phi$ collocation points: a dense $2n_\phi\times 2n_\phi$ system $Au = b$ whose four
blocks combine $D_\varphi$ with diagonal couplings. It is assembled from vectorized blocks
(no loop over rows). $Y_{2s}$ and $Y_{2c}$ follow from the area constraints.

{func}`~pyqsc_jax.second_order_residuals` evaluates all four equations above directly from
the solved coefficients, independently of the matrix assembly. For section 5.1 of
{cite:t}`landreman2019highorder` at $n_\phi = 61$ the largest residual is
{{ ls51_r2_residual }}.

## Linear solve and its derivative

The matrix is LU-factored once (`jax.scipy.linalg.lu_factor`). The same factors serve the
primal solve, the transposed solve of reverse mode, and a Hager-Higham estimate of the
1-norm condition number {cite}`higham1988fortran`. SOLVAX's `linear_solve` attaches the
implicit rules, so derivatives differentiate $Au = b$, not the factorization. The
{class}`~pyqsc_jax.LinearSolveReport` records the absolute and relative residual, the
condition estimate (and the exact one only with `diagnostics=True`), and three flags:
`finite`, `converged` (relative residual $\le 10^{-11}$) and `well_conditioned` (estimate
$\le 10^{12}$). Derivatives are NaN when `converged` is false. The system becomes
ill-conditioned as $\iota_N\to 0$; check the report rather than expecting a warning from
compiled code. Section 5.1 gives a condition estimate of {{ ls51_linear_condition }}.

## $B_{20}$ and $G_2$

$$
B_{20} = B_0\left[\kappa X_{20} - \frac{Z_{20}'}{\ell'} + \frac{\bar\eta^2}{2}
 - \frac{\mu_0 p_2}{B_0^2} - \frac{q_c^2 + q_s^2 + r_c^2 + r_s^2}{4\ell'^2}\right],
\qquad
G_2 = -\frac{\mu_0 p_2 G_0}{B_0^2} - \iota I_2.
$$

Quasisymmetry at second order needs $B_{20}$ constant; its variation measures the
violation. The solution stores the arclength-weighted mean `B20_mean`, the anomaly
$B_{20} - \langle B_{20}\rangle$, `B20_residual`
$=\langle(B_{20} - \langle B_{20}\rangle)^2\rangle^{1/2}/B_0$, and
`B20_variation` $=\max B_{20} - \min B_{20}$. It also stores the $\varphi$ derivatives of
all coefficients and the untwisted forms used by the boundary.

For section 5.1: $\iota =$ {{ ls51_iota }}, $\langle B_{20}\rangle =$ {{ ls51_B20_mean }}
T/m², $\max B_{20} - \min B_{20} =$ {{ ls51_B20_variation }} T/m².

```{figure} ../_static/figures/convergence.png
:width: 100%
:alt: Spectral convergence of iota, B20, the field-jet divergence and the axis integral

Convergence with the number of grid points per period, against $n_\phi =$ {{ convergence_reference_nphi }}. (a) $\iota$ and $B_{20}(0)$ for section 5.1: spectral
decay to about $10^{-15}$ and $10^{-11}$. (b) $\max|\nabla\cdot\mathbf B|$ of the regular-coordinate field
jet ({doc}`field_jet`). (c) The regularized axis integral of {doc}`plasma_field` for
database configuration 52521, with and without the kink correction ($h^4$ versus $h^2$).
```
