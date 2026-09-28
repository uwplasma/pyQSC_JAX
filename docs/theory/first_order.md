# First order

## The sigma equation

At first order the cross-section is an ellipse whose shape is fixed by $\bar\eta$ and the
axis curvature, and whose rotation is described by a periodic function $\sigma(\varphi)$.
Quasisymmetry at $O(r)$ requires {cite}`garren1991existence,landreman2019numerical`

$$
\frac{d\sigma}{d\varphi}
+ \iota_N\left[\frac{\bar\eta^4}{\kappa^4} + 1 + \sigma^2\right]
- 2\,\frac{\bar\eta^2}{\kappa^2}\left(\frac{I_2}{B_0} - s_\psi\tau\right)\frac{G_0}{B_0} = 0,
\qquad \sigma(0) = \sigma_0,
$$

with $G_0/B_0 = s_G\ell'$. For given $\bar\eta$, $I_2$ and axis, this is a periodic Riccati
equation whose unknowns are the function $\sigma$ and the number $\iota$.

## Discretization and Newton solve

The state vector has $n_\phi$ entries: $\iota$ in slot 0 and $\sigma(\varphi_j)$ for
$j = 1,\dots,n_\phi-1$; slot 0 of $\sigma$ is replaced by the prescribed $\sigma_0$, which
removes the one redundant degree of freedom. The residual is the equation above at every
grid point, with $d/d\varphi$ the spectral matrix $D_\varphi$.

`solvers.dense_newton_root` solves it with damped Newton:

1. start from $\iota = 0$, $\sigma = \sigma_0$;
2. form the exact Jacobian with `jax.jacfwd` and solve for the full step;
3. halve the step (at most `max_backtracking_steps` times) until the infinity norm of the
   residual decreases; only a decreasing step is accepted, so the returned iterate is the
   best one seen, and a failed line search stops the iteration;
4. stop when $\|F\|_\infty \le \max(\mathrm{atol}, \mathrm{rtol}\,\|F_0\|_\infty)$, when the
   step is below `step_tolerance`$\,(1 + \|x\|_\infty)$, or after `max_steps`.

The defaults in {class}`~pyqsc_jax.RootSolveOptions` are
$\mathrm{atol} = \mathrm{rtol} = 10^{-13}$, 20 steps and 12 halvings; the QA example
converges in {{ qa_newton_iterations }} steps. The {class}`~pyqsc_jax.RootSolveReport` holds
the initial and final residual, the tolerance used, the last step, the iteration and
backtracking counts, `converged`, `finite`, `stagnated` and `line_search_failed`, and a
Hager-Higham 1-norm condition estimate of the final Jacobian from its LU factors
{cite}`higham1988fortran` ($O(n^2)$). The exact 2-norm condition number (an SVD) is
computed only with `exact_condition_number=True` or `solve(..., diagnostics=True)`.
Non-convergence is reported, not raised: the candidate is returned with
`converged = False`.

## Implicit differentiation

The Newton candidate is passed through `stop_gradient` to SOLVAX's `root_solve`, which
attaches the implicit-function rule

$$
F(x^\star, p) = 0\quad\Rightarrow\quad
\frac{dx^\star}{dp} = -\left(\frac{\partial F}{\partial x}\right)^{-1}\frac{\partial F}{\partial p},
$$

for both JVPs and VJPs. The Newton iterations themselves are never differentiated. The
derivative is valid only at a converged root with a regular Jacobian, so when
`converged` is false or the condition estimate is not finite every tangent and cotangent of
the root is set to NaN ({doc}`../user_guide/differentiation`).

## Shape, field and gradient

With the Frenet-frame coefficients

$$
X_{1s} = 0,\qquad X_{1c} = \frac{\bar\eta}{\kappa},\qquad
Y_{1s} = \frac{s_G s_\psi\,\kappa}{\bar\eta},\qquad Y_{1c} = Y_{1s}\,\sigma,
$$

the elongation of the ellipse in the plane normal to the axis is

$$
e = \frac{p + \sqrt{p^2 - 4q^2}}{2|q|},\qquad
p = X_{1s}^2 + X_{1c}^2 + Y_{1s}^2 + Y_{1c}^2,\qquad q = X_{1s}Y_{1c} - X_{1c}Y_{1s},
$$

and `mean_elongation` is its arclength average. The on-axis field is
$\mathbf B_0 = s_G B_0\,\mathbf t$ and its gradient is

$$
\nabla\mathbf B = \sum_{a,b\in\{t,n,b\}} c_{ab}\,\mathbf e_a\mathbf e_b,
$$

with the first index the derivative direction and, writing $\Lambda = s_\psi B_0^2/|G_0|$ and
a prime for $d/d\varphi$,

$$
\begin{aligned}
c_{tn} &= c_{nt} = s_G B_0\kappa,\qquad c_{tt} = 0,\\
c_{nn} &= \Lambda\left(X_{1c}'Y_{1s} + \iota_N X_{1c}Y_{1c}\right),\qquad
c_{bb} = \Lambda\left(X_{1c}Y_{1s}' - \iota_N X_{1c}Y_{1c}\right),\\
c_{bn} &= \Lambda\left(-s_Gs_\psi\ell'\tau - \iota_N X_{1c}^2\right),\\
c_{nb} &= \Lambda\left(Y_{1c}'Y_{1s} - Y_{1s}'Y_{1c} + s_Gs_\psi\ell'\tau + \iota_N(Y_{1s}^2 + Y_{1c}^2)\right).
\end{aligned}
$$

`grad_B_axis[n, i, j]` $= \partial B_i/\partial x_j$ is this tensor in Cartesian components
(`grad_B_axis_cylindrical` in cylindrical ones), and

$$
L_{\nabla B} = B_0\sqrt{\frac{2}{\nabla\mathbf B : \nabla\mathbf B}}
$$

is the gradient scale length {cite}`landreman2021figures`. In vacuum the tensor is
symmetric and trace-free to spectral accuracy.

## Reference values

For the QA axis $R = 1 + 0.045\cos3\phi$, $Z = -0.045\sin3\phi$, $\bar\eta = -0.9$,
$n_\phi = 31$: $\iota =$ {{ qa_iota }} and $L =$ {{ qa_axis_length }} m. Sigma, $\iota$,
elongation, field and gradient agree with pyQSC ({doc}`../development/index`).
