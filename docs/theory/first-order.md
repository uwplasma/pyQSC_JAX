# First-order quasisymmetric construction

## Inputs and signs

The first-order solve uses the direct-construction conventions of
{cite}`landreman2019highorder`. The leading field strength is `B0`; `sG` is
the sign of \(G_0\), and `spsi` is the sign of toroidal flux. The axis geometry
provides \(\kappa\), \(\tau\), \(\varphi\), and

\[
\frac{|G_0|}{B_0} = \frac{L}{2\pi}.
\]

`frame_helicity` is the signed winding of the Frenet normal over one field
period. The sign-adjusted helicity and transform entering the sigma equation
are

\[
N = h\,s_\psi s_G,
\qquad
\iota_N = \iota + N n_{\mathrm{fp}}.
\]

This convention gives \(N=0\) for the standard QA example and \(N=-1\) for
the documented four-field-period QH example when `spsi = sG = 1`.

## Periodic sigma equation

For prescribed \(\bar\eta\), the unknown state contains the rotational
transform in its first slot and samples of the periodic function
\(\sigma(\varphi)\) in the remaining slots. The first sigma sample is replaced
by the prescribed `sigma0`, which removes the otherwise redundant degree of
freedom. The collocation residual is

\[
\frac{\mathrm d\sigma}{\mathrm d\varphi}
+ \iota_N
\left[
1 + \sigma^2
+ \left(\frac{\bar\eta^2}{\kappa^2}\right)^2
\right]
-2\frac{\bar\eta^2}{\kappa^2}
\left(-s_\psi\tau + \frac{I_2}{B_0}\right)
\frac{s_G|G_0|}{B_0}
=0.
\]

Fourier collocation differentiates the periodic array. A damped dense Newton
solve uses the exact JAX Jacobian, residual-based backtracking, a step
stagnation test, and configurable absolute/relative tolerances. It returns a
`RootSolveReport` containing:

- initial and final infinity-norm residuals;
- the effective residual tolerance and final step norm;
- Newton and accumulated backtracking counts;
- final Jacobian condition number;
- convergence, finiteness, and stagnation flags.

Reaching an iteration or stagnation limit does not silently certify a
solution. The numerical candidate and a false convergence flag are returned
so batched workflows can apply their own rejection policy.

## Implicit differentiation

The converged candidate is supplied to SOLVAX's custom root wrapper after
`stop_gradient`. JVPs and VJPs therefore differentiate

\[
F(x,p)=0,
\qquad
\frac{\mathrm dx}{\mathrm dp}
=
-\left(\frac{\partial F}{\partial x}\right)^{-1}
\frac{\partial F}{\partial p},
\]

not the sequence of Newton iterates. Tests compare JVP and VJP, finite
differences in \(\bar\eta\) and an axis Fourier coefficient, eager/JIT
evaluation, and VMAP batches.

## First-order surface and field jet

The Frenet-plane coefficients are

\[
X_{1s}=0,\qquad
X_{1c}=\frac{\bar\eta}{\kappa},\qquad
Y_{1s}=\frac{s_Gs_\psi\kappa}{\bar\eta},\qquad
Y_{1c}=\sigma Y_{1s}.
\]

They are also provided in an untwisted frame for boundary conversion. The
solution contains the on-axis field in cylindrical and Cartesian bases, the
Cartesian and cylindrical field-gradient tensors, elongation, and

\[
L_{\nabla B}
= B_0\sqrt{\frac{2}{\nabla\boldsymbol B:\nabla\boldsymbol B}}.
\]

Canonical arrays use a leading sample axis. Thus `B_axis.shape ==
(nphi, 3)` and `grad_B_axis.shape == (nphi, 3, 3)`, with
`grad_B_axis[n, i, j] = d B_i / d x_j`. The ESSOS adapter preserves the
historical component-axis layout. In the vacuum limit, resolution studies
verify the trace-free and symmetric gradient identities spectrally.

## Reference configuration

For

```python
import pyqsc_jax as qsc

solution = qsc.Qsc(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
    etabar=-0.9,
    nphi=31,
)
```

the validated result is

\[
\iota = 0.41830690943386617,
\qquad
L = 6.340238817434161.
\]

The sigma array, QA/QH topology, finite-current result, on-axis field, and
field gradient agree with pyQSC at the audited upstream revision recorded in
the refactor baseline.
