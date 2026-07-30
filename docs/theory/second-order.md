# Finite-pressure/current second order

## Scope

Setting `order="r2"` computes the complete second-order surface coefficients
and \(B_{20}\) system of {cite}`landreman2019highorder`. The implementation
follows the staged dependencies in manuscript equations 36–54 and the
independently audited pyQSC implementation, while replacing mutable row-wise
assembly with pure JAX array operations.

The pressure and current inputs use

\[
p = p_0 + r^2 p_2 + O(r^4),
\qquad
I = r^2 I_2 + O(r^4).
\]

Both may be nonzero. `B2c` and `B2s` prescribe the constant second-harmonic
field-strength coefficients.

## Algebraic stages

Define

\[
V_1=X_{1c}^2+Y_{1c}^2+Y_{1s}^2,\qquad
V_2=2Y_{1s}Y_{1c},\qquad
V_3=X_{1c}^2+Y_{1c}^2-Y_{1s}^2.
\]

The tangent-direction coefficients are obtained directly:

\[
Z_{20}=-\frac{B_0}{8|G_0|}V_1',
\]

\[
Z_{2s}=-\frac{B_0}{8|G_0|}
\left(V_2'-2\iota_N V_3\right),
\qquad
Z_{2c}=-\frac{B_0}{8|G_0|}
\left(V_3'+2\iota_N V_2\right),
\]

where a prime means \(\mathrm d/\mathrm d\varphi\). These quantities determine
\(X_{2s}\) and \(X_{2c}\) algebraically from the second-harmonic
field-strength equations.

The pressure response is

\[
\beta_{1s}
=-\frac{4s_\psi s_G\mu_0 p_2\bar\eta |G_0|}
{\iota_N B_0^3}.
\]

The two remaining periodic unknowns are stacked as

\[
u=(X_{20},Y_{20})
\]

and satisfy a dense collocation system \(A u=b\). `A` contains the Fourier
derivative blocks and pointwise couplings; it is assembled from four
vectorized blocks, with no Python loop over grid rows. After the solve,
\(Y_{2s}\) and \(Y_{2c}\) follow from the two area constraints.

## Implicit linear differentiation

The primal callback uses `jax.numpy.linalg.solve`. SOLVAX wraps the matrix
action, primal solve, transpose action, and transpose solve with a custom
linear solve. JVPs and VJPs therefore differentiate the equation \(Au=b\),
not the internals of the dense factorization.

`linear_report` records:

- absolute and relative infinity-norm residuals;
- matrix condition number;
- finite, converged, and well-conditioned flags.

The default report marks condition numbers above \(10^{12}\) as poorly
conditioned. Small \(|\iota_N|\) is a known physical conditioning risk and
must be assessed through this report rather than a warning emitted inside
JIT-compiled code.

## \(B_{20}\) and direct diagnostics

The nonconstant second-order field strength is

\[
B_{20}=B_0\left[
\kappa X_{20}
-\frac{B_0}{|G_0|}Z_{20}'
+\frac{\bar\eta^2}{2}
-\frac{\mu_0p_2}{B_0^2}
-\frac{B_0^2}{4|G_0|^2}
\left(q_c^2+q_s^2+r_c^2+r_s^2\right)
\right],
\]

with the standard first-order combinations \(q_c,q_s,r_c,r_s\). The result
includes the axis-length-weighted mean, anomaly, normalized weighted
\(L^2\) residual, and peak-to-peak variation. It also provides

\[
G_2=-\frac{\mu_0p_2G_0}{B_0^2}-\iota I_2.
\]

All second-order coefficient derivatives needed by the next field-tensor and
diagnostic milestone are stored. Untwisted zero- and second-harmonic
coefficients are used by the ESSOS-compatible boundary conversion.

## Validation

Four equations are evaluated independently of matrix assembly: two coupled
force-balance equations and two area constraints. Their maximum residual is
tested directly. Upstream regression cases cover:

- the vacuum QA configuration of section 5.1;
- the finite-pressure/current configuration of section 5.3;
- the QH configuration of section 5.4;
- an asymmetric-axis development comparison.

JIT, VMAP, JVP/VJP consistency, finite differences, and legacy r2 boundary
availability are also tested. Reference values are tied to the audited pyQSC
commit recorded in the refactor baseline.

This milestone does not yet expose the total-field Hessian, Mercier criterion,
or singular-radius diagnostics. Those are separate validation gates and are
not implied by `order="r2"`.
