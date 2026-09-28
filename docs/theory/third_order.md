# The pyQSC-compatible third-order correction

## Scope

`order="r3"` does **not** solve the third-order equilibrium equations. It adds only the
correction that pyQSC calls `r3_flux_constraint` {cite}`landreman2019highorder`: a
first-harmonic $O(r^3)$ displacement that keeps the toroidal flux enclosed by the
constructed surface consistent through $O(r^2)$. It changes the boundary written to VMEC
and drawn by the plotting helpers; it changes nothing on the axis (field, gradient,
Hessian, $\iota$, $B_{20}$). Magnetic shear, the third-order field strength and the
$\cos3\vartheta$, $\sin3\vartheta$ shaping are not computed; their arrays exist and are
zero, so boundary code has one uniform representation.

## The correction

Every nonzero coefficient is proportional to the first-order shape,

$$
X_{3c1} = X_{1c}\,C,\qquad Y_{3s1} = Y_{1s}\,C,\qquad Y_{3c1} = Y_{1c}\,C,
\qquad X_{3s1} = Z_{3c1} = Z_{3s1} = 0,
$$

with a periodic function $C(\varphi)$. The primal value of $C$ is the parent $O(r^3)$
expression of pyQSC's `calculate_r3.py`, transcribed term by term (a polynomial in the
first- and second-order coefficients divided by $16B_0^2G_0X_{1c}^2Y_{1s}^2$; see
`third_order._flux_constraint`). Two shorter forms that share no algebra with it are
evaluated as checks:

1. the regular-coordinate flux relation $C = -Q/(2s_Gs_\psi)$, with

   $$
   \begin{aligned}
   Q = {}& -\frac{s_\psi B_0\ell'}{2G_0^2}\left(\iota_N I_2 + \frac{\mu_0 p_2 G_0}{B_0^2}\right)
     + 2\left(X_{2c}Y_{2s} - X_{2s}Y_{2c}\right)\\
   & + \frac{s_\psi B_0}{2G_0}\left(\ell'\kappa X_{20} - Z_{20}'\right)\\
   & + \frac{I_2}{4G_0}\left[-\ell'\tau\left(X_{1c}^2 + Y_{1s}^2 + Y_{1c}^2\right)
     + Y_{1c}X_{1c}' - X_{1c}Y_{1c}'\right];
   \end{aligned}
   $$

2. the on-axis field-strength cancellation $C = B^{(2)}_{0,\mathrm{cancel}}/(2B_0)$, with

   $$
   \begin{aligned}
   B^{(2)}_{0,\mathrm{cancel}} = {}& -\frac{s_G B_0^2\ell'}{2G_0^2}\left(G_2 + N_h I_2\right)
     - 2s_Gs_\psi B_0\left(X_{2c}Y_{2s} - X_{2s}Y_{2c}\right)\\
   & - \frac{s_G B_0^2}{2G_0}\left(\ell'\kappa X_{20} - Z_{20}'\right)\\
   & - \frac{s_Gs_\psi B_0 I_2}{4G_0}\left[-\ell'\tau\left(X_{1c}^2 + Y_{1c}^2 + Y_{1s}^2\right)
     + Y_{1c}X_{1c}' - X_{1c}Y_{1c}'\right],
   \end{aligned}
   $$

   where $N_h = \iota - \iota_N = -Nn_\mathrm{fp}$.

`flux_constraint_residual` $=\max|C + Q/(2s_Gs_\psi)|$ and `consistency_error`
$=\max|C - B^{(2)}_{0,\mathrm{cancel}}/(2B_0)|$. They are the two quantities pyQSC warns
about. Both vanish only as the second-order solution is resolved, so they measure the
resolution: for section 5.1 they are {{ ls51_r3_flux_residual_61 }} and
{{ ls51_r3_consistency_61 }} at $n_\phi = 61$ and {{ ls51_r3_flux_residual_121 }} and
{{ ls51_r3_consistency_121 }} at $n_\phi = 121$.

The coefficients are rotated to the untwisted frame with the first-harmonic angle
$-Nn_\mathrm{fp}\varphi$. This matters for QH axes: a coefficient that vanishes in the
winding Frenet frame can acquire a sine component after untwisting. The VMEC boundary
includes these terms times $r^3$.
