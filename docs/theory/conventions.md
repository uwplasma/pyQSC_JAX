# Coordinates and conventions

## Magnetic axis

The axis is a closed curve given by Fourier series in the cylindrical angle $\phi$ over one
of $n_\mathrm{fp}$ field periods {cite}`landreman2018direct`:

$$
\begin{aligned}
R(\phi) &= \sum_{n\ge 0}\left[R_{cn}\cos(n n_\mathrm{fp}\phi) + R_{sn}\sin(n n_\mathrm{fp}\phi)\right],\\
Z(\phi) &= \sum_{n\ge 0}\left[Z_{cn}\cos(n n_\mathrm{fp}\phi) + Z_{sn}\sin(n n_\mathrm{fp}\phi)\right].
\end{aligned}
$$

{class}`~pyqsc_jax.Axis` stores `rc, rs, zc, zs` padded to a common length; the packed
degree-of-freedom vector is ordered `rc, rs, zc, zs`. A stellarator-symmetric axis has
`rs = zc = 0`, which is not assumed by the geometry.

## Grid and Frenet frame

Every quantity is sampled on the endpoint-free grid of one field period,

$$
\phi_j = \frac{2\pi j}{n_\mathrm{fp} n_\phi},\qquad j = 0,\dots,n_\phi-1,
$$

and `nphi` is used as given (even values are not changed). The first three derivatives of
$\mathbf r_0(\phi)$ are analytic. With $\ell$ the arclength,

$$
\mathbf t = \frac{d\mathbf r_0}{d\ell},\qquad
\kappa\,\mathbf n = \frac{d\mathbf t}{d\ell},\qquad
\mathbf b = \mathbf t\times\mathbf n,\qquad
\tau = \frac{\mathbf r_0'\cdot(\mathbf r_0''\times\mathbf r_0''')}{|\mathbf r_0'\times\mathbf r_0''|^2}.
$$

The torsion sign follows Landreman and Sengupta, opposite to the original Garren-Boozer
papers {cite}`garren1991magnetic,landreman2019highorder`. Cylindrical components are
ordered $(R,\phi,Z)$ and Cartesian components $(x,y,z)$; sampled vector arrays have shape
`(nphi, 3)` with the sample index first. `geometry.frenet_frame` stacks the rows
$(\mathbf t,\mathbf n,\mathbf b)$ in Cartesian components.

Samples with speed below $10^{-12}$ or curvature below $10^{-10}$ get NaN frame vectors and
torsion, and `geometry.diagnostics.frenet_valid` is false. Minimum speed, curvature,
cylindrical radius, frame orthogonality and frame determinant are reported in
`GeometryDiagnostics`.

## Boozer toroidal angle

The Boozer angle $\varphi$ satisfies

$$
\frac{d\varphi}{d\phi} = \frac{B_0}{|G_0|}\frac{d\ell}{d\phi},\qquad
\frac{|G_0|}{B_0} = \ell' = \frac{L}{2\pi},\qquad \varphi(0) = 0,
$$

with $L$ the axis length, so $\varphi$ advances by $2\pi/n_\mathrm{fp}$ per period.
`geometry.varphi` is the cumulative trapezoidal integral (as in pyQSC). Derivatives with
respect to $\varphi$ use the Fourier collocation matrix $D_\phi$ of Weideman and Reddy
{cite}`weideman2000matlab`, $D_\varphi = \mathrm{diag}(d\varphi/d\phi)^{-1}D_\phi$
(`geometry.d_d_varphi`); they are spectrally accurate for periodic functions.

## Signs, helicity and the helical angle

| symbol | code | meaning |
|---|---|---|
| $s_G$ | `sG` | sign of $G_0$ (the covariant toroidal field), $G_0 = s_G B_0 L/2\pi$ |
| $s_\psi$ | `spsi` | sign of the toroidal flux |
| $\chi$ | `sG * spsi` | orientation factor used in the current conversions |
| $h$ | `geometry.frame_helicity` | signed number of turns of $\mathbf n$ about the axis per period |
| $N$ | `solution.helicity` | $N = h\,s_\psi s_G$ |
| $\iota_N$ | `solution.iotaN` | $\iota_N = \iota + N n_\mathrm{fp}$ |

$h$ is counted from quadrant crossings of the $(n_R, n_Z)$ components of the normal. It is
zero for quasi-axisymmetry (QA) and nonzero for quasi-helical symmetry (QH); for the
four-period QH example of {cite:t}`landreman2019numerical` it gives $N =$ {{ qh_helicity }},
$\iota =$ {{ qh_iota }} and $\iota_N =$ {{ qh_iotaN }}. The expansion uses the helical angle
$\vartheta = \theta - N n_\mathrm{fp}\varphi$, in which the field strength is

$$
B(r,\vartheta,\varphi) = B_0\left[1 + r\bar\eta\cos\vartheta\right]
 + r^2\left[B_{20}(\varphi) + B_{2c}\cos 2\vartheta + B_{2s}\sin 2\vartheta\right] + O(r^3),
$$

where $r = \sqrt{2|\psi|/B_0}$ and $\psi$ is the toroidal flux over $2\pi$. The surface is

$$
\mathbf x(r,\vartheta,\varphi) = \mathbf r_0 + X\,\mathbf n + Y\,\mathbf b + Z\,\mathbf t,\qquad
X = rX_1 + r^2X_2 + r^3X_3 + \dots
$$

with $X_1 = X_{1c}\cos\vartheta + X_{1s}\sin\vartheta$ and
$X_2 = X_{20} + X_{2c}\cos2\vartheta + X_{2s}\sin2\vartheta$, and likewise for $Y$ and $Z$.
The `*_untwisted` arrays are the same coefficients rotated to the Boozer poloidal angle
$\theta$, by the angle $-N n_\mathrm{fp}\varphi$ times the harmonic number; the boundary and
VMEC export use them.

## Pressure and current

$$
p(r) = p_0 + r^2 p_2 + O(r^4),\qquad I(r) = r^2 I_2 + O(r^4),\qquad G(r) = G_0 + r^2G_2 + O(r^4),
$$

where $I$ is the covariant poloidal field (toroidal current) in the pyQSC normalization.
`p2` is usually negative (pressure falling outwards).

## Tensor layout

| array | shape | element |
|---|---|---|
| `B_axis` | `(nphi, 3)` | $B_i$ |
| `grad_B_axis` | `(nphi, 3, 3)` | $\partial B_i/\partial x_j$ |
| `grad_grad_B_axis` | `(nphi, 3, 3, 3)` | $\partial^2 B_i/\partial x_j\partial x_k$ |

The field component comes first. pyQSC's historical $(\mathbf n,\mathbf b,\mathbf t)$
Hessian with derivative indices first is `field_jet.hessian_frenet`; the ESSOS adapter
{class}`pyqsc_jax.near_axis.near_axis` keeps the historical component-first, sample-last
layouts.
