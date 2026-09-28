# Third-order flux constraint

## Scope

Setting `order="r3"` first constructs the complete r2 solution, then adds the
third-order surface displacement required for consistency of the toroidal-flux
constraint through order \(r^2\). This is the `r3_flux_constraint` construction
of {cite}`landreman2019highorder`.

The implementation uses the short regular-coordinate relation as its primal
formula. It also evaluates the independent field-strength cancellation
formula retained by pyQSC; their agreement is reported instead of assumed.

## Flux constraint

Let \(\ell'=|G_0|/B_0\), let a prime denote
\(\mathrm d/\mathrm d\varphi\), and define

\[
\begin{aligned}
Q={}&-\frac{s_\psi B_0\ell'}{2G_0^2}
\left(\iota_N I_2+\frac{\mu_0p_2G_0}{B_0^2}\right)
+2(X_{2c}Y_{2s}-X_{2s}Y_{2c})\\
&+\frac{s_\psi B_0}{2G_0}
\left(\ell'\kappa X_{20}-Z_{20}'\right)\\
&+\frac{I_2}{4G_0}
\left[
-\ell'\tau(X_{1c}^2+Y_{1s}^2+Y_{1c}^2)
+Y_{1c}X_{1c}'-X_{1c}Y_{1c}'
\right].
\end{aligned}
\]

The scalar coefficient at every toroidal sample is

\[
C=-\frac{Q}{2s_Gs_\psi}.
\]

For the quasisymmetric flux-constraint construction, the nonzero Frenet-frame
coefficients are

\[
X_{3c1}=X_{1c}C,\qquad
Y_{3s1}=Y_{1s}C,\qquad
Y_{3c1}=Y_{1c}C.
\]

\(X_{3s1}\), all \(Z_{3}\) coefficients, and all third-poloidal-harmonic
coefficients vanish in this construction. They remain explicit arrays in
`ThirdOrderData` so boundary assembly has a uniform representation and can be
extended without changing its interface.

## Independent consistency checks

The separately evaluated on-axis field correction must satisfy

\[
B_{0,\mathrm{cancel}}^{(2)}=2B_0C.
\]

`flux_constraint_residual` is the maximum absolute residual of
\(Q+2s_Gs_\psi C=0\). `consistency_error` is the maximum disagreement between
\(C\) and \(B_{0,\mathrm{cancel}}^{(2)}/(2B_0)\). Both are scalar JAX arrays
and remain available inside compiled and differentiated workflows.

## Helicity and boundary data

The coefficient pairs are rotated from the winding Frenet frame into the
untwisted frame for first and third poloidal harmonics. The
ESSOS-compatible boundary map includes these terms multiplied by \(r^3\).
This is essential for QH axes: a coefficient that vanishes in the winding
frame can acquire a nonzero sine component after untwisting.

## Validation

Frozen upstream comparisons use the vacuum QA, finite-pressure/current, and QH
configurations of sections 5.1, 5.3, and 5.4 of
{cite}`landreman2019highorder`. Tests also cover both constraint residuals,
spectral coefficient derivatives, helical untwisting, r3 boundary assembly,
JIT, JVP, and centered finite differences. The upstream commit is recorded in
the refactor baseline.

## Magnetic shear

The generalized sigma equation gives the next rotational-transform term

\[
\iota(r)=\iota_0+r^2\iota_2+O(r^4)
\]

as a periodic solvability condition {cite}`rodriguez2022weakly`. The source
derivation uses \(\epsilon=\sqrt{\psi}\), whereas the direct-construction
variables use \(r=\sqrt{2\psi/B_0}\). `solve_magnetic_shear` performs this
conversion explicitly before evaluating the third-order \(Z_{31}\),
\(X_{31}\), and \(Y_{31}\) relations.

Writing \(\widetilde{\Lambda}\) for the inhomogeneous term and

\[
\mathcal E(\varphi)=
\exp\left(2\iota_N\int_0^\varphi\sigma(v)\,\mathrm dv\right),
\]

the stellarator-symmetric result has the quotient form

\[
\iota_2=
\frac{B_0}{2}
\frac{\int \mathcal E\,\widetilde{\Lambda}\,\mathrm d\phi}
{\int \mathcal E
(X_{1c}^2+Y_{1c}^2+Y_{1s}^2)Y_{1s}^{-2}\,\mathrm d\phi}.
\]

For a general asymmetric first-order solution, \(\sigma\) can have a nonzero
mean. The implementation separates the periodic and secular parts before
forming \(\mathcal E\), appends the analytically consistent endpoint, and uses
trapezoidal integration over one field period. This avoids applying a periodic
inverse derivative to a nonperiodic primitive.

`B31c` is an explicit scalar input in the inverse-\(B^2\) convention.
`ShearData` records the numerator, denominator, integrating factor,
\(\widetilde{\Lambda}\), and third-order intermediates as well as `iota2`.
The current implementation is the standard-MHS specialization with
\(B_{31s}=0\), \(I_4=0\), and \(s_G=s_\psi=1\); other sign conventions are
rejected rather than silently extrapolated.

Validation covers upstream pyQSC values for the section 5.1 vacuum QA,
section 5.3 finite-pressure/current, and section 5.4 QH configurations; an
asymmetric axis and nonzero sigma mean; resolution convergence; `B31c`
dependence; JIT, JVP, VJP, and differentiation through the complete near-axis
solve.
