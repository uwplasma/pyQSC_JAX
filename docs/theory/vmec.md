# VMEC export

## Boundary on a uniform cylindrical grid

A near-axis surface at radius $r$ is naturally parametrized by the axis angle $\phi_0$ of
its foot point, not by the cylindrical angle $\phi$ that VMEC {cite}`hirshman1983vmec` uses.
For each $(\theta, \phi_0)$ the displacement
$X\mathbf n + Y\mathbf b + Z\mathbf t$ (untwisted coefficients through the solved order,
including the $r^3$ correction) is added to the axis point, periodically interpolated in
$\phi_0$, and converted to

$$
R = \sqrt{(R_0 + \Delta_R)^2 + \Delta_\phi^2},\qquad
\phi = \phi_0 + \operatorname{atan2}\left(\Delta_\phi, R_0 + \Delta_R\right),\qquad
Z = Z_0 + \Delta_Z .
$$

{func}`~pyqsc_jax.uniform_cylindrical_surface` solves $\phi(\theta,\phi_0) = \phi_j$ for
$\phi_0$ on the whole $(\theta, \phi)$ grid at once with 6 Newton steps; the derivative
$\partial\phi/\partial\phi_0$ comes from a JVP. {func}`~pyqsc_jax.vmec_boundary` reports
the largest angle residual against a tolerance of $100\,\epsilon_\mathrm{mach}$ by default,
and a convergence flag (a value, so the function stays jittable).

## Fourier coefficients

A two-dimensional FFT of $R$ and $Z$ on the $n_\theta\times n_\phi$ grid gives

$$
R = \sum_{m=0}^{m_\mathrm{pol}}\sum_{n=-n_\mathrm{tor}}^{n_\mathrm{tor}}
\left[R^{bc}_{mn}\cos(m\theta - nn_\mathrm{fp}\phi) + R^{bs}_{mn}\sin(m\theta - nn_\mathrm{fp}\phi)\right],
$$

and likewise $Z$ with $Z^{bc}$, $Z^{bs}$. All four families are computed, so asymmetric
axes are supported; `LASYM = T` is written when any $R^{bs}$ or $Z^{bc}$ exceeds the
coefficient tolerance. The surface is rebuilt from the retained modes and the maximum $R$
and $Z$ reconstruction errors are reported. Arrays have shape `(2 ntor + 1, mpol + 1)`,
toroidal index from $-n_\mathrm{tor}$ to $n_\mathrm{tor}$. The resolutions must satisfy
$n_\theta\ge 2(m_\mathrm{pol} + 1)$ and $n_\phi\ge 2n_\mathrm{tor} + 1$.

## The `&INDATA` file

{func}`~pyqsc_jax.to_vmec` refuses to write a file if the angle inversion did not
converge, then writes:

| entry | value |
|---|---|
| `MPOL` | `mpol + 1` |
| `NTOR`, `NFP` | `ntor`, $n_\mathrm{fp}$ |
| `PHIEDGE` | $\pi r^2B_0$ |
| `AM` (power series) | $(-p_2r^2,\ p_2r^2)$, i.e. $p(s) = -p_2r^2(1 - s)$ with $s$ the normalized flux |
| `CURTOR`, `NCURR = 1`, `AC = 1` | $2\pi I_2r^2/\mu_0$ |
| `RAXIS_CC`, `RAXIS_CS` | `rc`, `-rs` |
| `ZAXIS_CC`, `ZAXIS_CS` | `zc`, `-zs` |
| boundary | every `RBC`, `ZBS` (and `RBS`, `ZBC` if `LASYM`) above the tolerance |

plus the numerical controls of {class}`~pyqsc_jax.VmecInputParameters`
(`DELT`, `NSTEP`, `TCON0`, `NS_ARRAY`, `FTOL_ARRAY`, `NITER_ARRAY`). The header comments
record the radius, order, resolution and conversion diagnostics.

**MPOL convention.** `mpol` is the highest poloidal index in the boundary. VMEC keeps modes
$m < \mathrm{MPOL}$, so the file has `MPOL = mpol + 1`. pyQSC writes `MPOL = mpol`, and VMEC
then silently drops its $m = \mathrm{mpol}$ row. The same convention holds for
{func}`~pyqsc_jax.to_vmex_problem`.

**Sign of the toroidal angle.** The axis sine coefficients are negated because VMEC's
toroidal angle runs opposite to the one used here; VMEC therefore reports $\iota$ with the
opposite sign. `VmexRadialQuantities.iota` is converted back to the pyQSC_JAX sign and
`iota_vmec` keeps VMEC's.

## Scope

The export is a fixed-boundary input at a chosen radius. The on-axis $\iota$ of the
resulting equilibrium agrees with the near-axis value only asymptotically, with an error
that grows like $r^2$; VMEC2000 9.0 runs of the QA example, kept in the study repository
[plasma-coil-fields](https://github.com/rogeriojorge/plasma-coil-fields/tree/pyqsc-jax-validation/pyqsc_jax_validation), give relative differences of
0.06 % at $r = 0.0025$ m and 17 % at $r = 0.03$ m. The test suite rereads the frozen
$r = 0.0025$ m `wout` file on every run. Agreement at small
radius does not make a truncated boundary accurate far from the axis.
