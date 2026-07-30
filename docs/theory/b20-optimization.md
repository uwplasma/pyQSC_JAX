# \(B_{20}\) diagnostics and analytic \(B_{2c}\)

## Primary residual

The optimization target is the nonconstant part of \(B_{20}\), not its mean.
For positive quadrature weights \(w_j\), define

\[
(PB_{20})_j=B_{20,j}
-\frac{\sum_k w_kB_{20,k}}{\sum_k w_k}.
\]

`weighted_l2` is

\[
\left[
\frac{\sum_j w_j(PB_{20})_j^2}{\sum_jw_j}
\right]^{1/2}\frac{1}{B_0}.
\]

`b20_diagnostics` also returns a smooth high-\(p\) norm, the sampled maximum,
peak-to-peak variation, all nonzero toroidal Fourier coefficients, their norm,
and a high-mode tail ratio. The Fourier coefficients use direct weighted
quadrature in Boozer toroidal angle \(\varphi\), so they do not assume that the
uniform cylindrical-\(\phi\) samples are uniform in \(\varphi\).

## Exact affine elimination

For fixed axis and all other physical inputs, the complete second-order solve
is affine in \(B_{2c}\):

\[
B_{20}(\varphi;B_{2c})=u(\varphi)+B_{2c}v(\varphi).
\]

The implementation obtains \(u\) and \(v\) from two complete implicit linear
solves at \(B_{2c}=0\) and \(B_{2c}=1\). It then evaluates

\[
B_{2c}^{\star}
=-\frac{\langle Pu,Pv\rangle_w}{\langle Pv,Pv\rangle_w}.
\]

This is the global minimizer in the one-dimensional affine subproblem whenever
\(\langle Pv,Pv\rangle_w>0\). `optimize_B2c` fully recomputes the r2 solution
at the optimum and, when present, recomputes r3 and magnetic shear so no stale
derived arrays survive. `affine_reconstruction_error` independently compares
that solution to \(u+B_{2c}^{\star}v\).

For a circular axis the projected response can vanish to roundoff because
changing \(B_{2c}\) changes only the constant mode. Such cases are marked
`degenerate`, retain the supplied \(B_{2c}\), and already have a vanishing
nonconstant residual.

## Independent resolution verification

`verify_B20_resolution` holds the physical candidate fixed and rebuilds it at

\[
n_\phi(m)=m(n_\phi-1)+1
\]

for each requested multiplier. The default `(1, 2, 4)` therefore doubles and
quadruples the number of grid intervals while retaining an odd grid. It reports
every dense diagnostic and successive relative changes. This check is
deliberately separate from the optimization; a small collocation objective is
not accepted without fine-grid verification.

## Validation

Tests establish:

- direct agreement of every scalar diagnostic with its definition;
- stationarity and strict improvement of the analytic optimum;
- machine-precision affine reconstruction;
- JIT and JVP agreement with finite differences;
- correct degenerate behavior for a circular axis;
- r3 and shear recomputation;
- convergence of weighted and maximum residuals under grid refinement.

Multistart axis optimization and basin enumeration build on these residuals but
remain a separate milestone.
