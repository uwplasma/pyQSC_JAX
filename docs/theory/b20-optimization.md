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
their \(L^1\) certificate, and a high-mode tail ratio. The Fourier coefficients
use direct weighted quadrature in Boozer toroidal angle \(\varphi\), so they do
not assume that the uniform cylindrical-\(\phi\) samples are uniform in
\(\varphi\).

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

The branch-aware multistart workflow built on these residuals is described in
{doc}`global-search`.

## Optimizer comparison and selected case

Several optimizers were tested from the Landreman--Paul precise-QA axis. For
the common comparison, Fourier modes 1--3 were varied and \(B_{2c}\) was
eliminated exactly at every objective evaluation. Timings are warm local
measurements and exclude the shared JAX compilation.

| method | evaluations | time [s] | independently evaluated weighted \(L^2\) |
| --- | ---: | ---: | ---: |
| exact \(B_{2c}\) only | 1 | — | \(3.7483\times10^{-4}\) |
| SciPy L-BFGS-B | 72 | 0.146 | \(1.9231\times10^{-4}\) |
| SciPy `least_squares` | 60 | 2.743 | \(1.9146\times10^{-4}\) |
| low-budget differential evolution | 120 | 0.137 | \(1.0281\times10^{-1}\) |
| pyQSC_JAX multistart Levenberg--Marquardt | 129 | 20.766* | \(1.2441\times10^{-4}\) |

The SciPy timings exclude the shared JAX residual/Jacobian compilation; the
asterisked multistart timing includes compilation of its independent closure.
The coarse differential-evolution budget is included to show why a method
label alone does not establish global quality. It did not locate the narrow
good basin. The selected `b20_optimized_qa` case used bounded
`least_squares` on modes 1--5 after exact \(B_{2c}\) elimination, followed by
independent resolution checks:

| `nphi` | weighted \(L^2\) |
| ---: | ---: |
| 61 | \(1.5900649432\times10^{-6}\) |
| 121 | \(1.5900649311\times10^{-6}\) |
| 241 | \(1.5900649186\times10^{-6}\) |
| 481 | \(1.5900649060\times10^{-6}\) |

At `nphi=121`, its dense maximum is \(3.20\times10^{-6}\) and peak-to-peak
variation is \(6.27\times10^{-6}\). This is 26,053 times smaller in weighted
residual than exact \(B_{2c}\) optimization of the stock README QA axis. The
computed singular radius is 0.2484 m. The README surface is drawn at
0.075 m, so it has a 3.31-fold radial clearance relative to that truncated-map
diagnostic. Both numbers appear next to the actual 3D surface; a flat
\(B_{20}\) curve is not presented without its geometry.

The result is a verified numerical basin, not a proof that no other basin is
better. The runnable comparison is
`benchmarks/benchmark_b20_optimizers.py`; its frozen report records the full
environment and raw values.
