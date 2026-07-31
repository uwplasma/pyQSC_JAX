# Axis coordinates and conventions

## Fourier axis

`Axis` represents a closed magnetic axis in cylindrical coordinates over one
field period:

\[
R(\phi) =
\sum_{n=0}^{N_F-1}
\left[
R_{cn}\cos(n n_{\mathrm{fp}}\phi)
+R_{sn}\sin(n n_{\mathrm{fp}}\phi)
\right],
\]

\[
Z(\phi) =
\sum_{n=0}^{N_F-1}
\left[
Z_{cn}\cos(n n_{\mathrm{fp}}\phi)
+Z_{sn}\sin(n n_{\mathrm{fp}}\phi)
\right].
\]

The arrays `rc`, `rs`, `zc`, and `zs` are padded with zeros to a common
one-dimensional shape `(nfourier,)`. The packed degree-of-freedom order is
exactly `rc, rs, zc, zs`. A stellarator-symmetric axis has `rs = zc = 0`.

```python
import pyqsc_jax as qsc

axis = qsc.Axis(
    rc=[1.0, 0.045],
    zs=[0.0, -0.045],
    nfp=3,
)
```

The representation follows the direct-construction convention
{cite}`landreman2018direct`. Unlike the historical pyQSC constructor, even
`nphi` is not silently changed; grid resolution is an explicit caller choice.

## Sampled geometry

`compute_axis_geometry(axis, nphi=...)` samples one field period on the
endpoint-free grid

\[
\phi_j = \frac{2\pi j}{n_{\mathrm{fp}}n_\phi},
\qquad j=0,\ldots,n_\phi-1.
\]

Analytic Fourier derivatives provide the first three derivatives of \(R\) and
\(Z\). Vector components in the cylindrical frame are ordered `(R, phi, Z)`;
Cartesian components are ordered `(x, y, z)`. Arrays of sampled vectors have
shape `(nphi, 3)`.

The tangent, normal, and binormal are

\[
\boldsymbol{t} = \frac{\mathrm{d}\boldsymbol{r}_0}{\mathrm{d}\ell},
\qquad
\boldsymbol{n} = \frac{1}{\kappa}
\frac{\mathrm{d}\boldsymbol{t}}{\mathrm{d}\ell},
\qquad
\boldsymbol{b} = \boldsymbol{t}\times\boldsymbol{n}.
\]

They form a right-handed orthonormal frame. Torsion uses the convention of
Landreman and Sengupta and the usual differential-geometry formula; this is
the opposite sign to the original Garren–Boozer convention
{cite}`garren1991magnetic,landreman2019highorder`.

The Boozer toroidal grid is normalized so that

\[
\frac{\mathrm{d}\varphi}{\mathrm{d}\phi}
= \frac{B_0}{|G_0|}
\frac{\mathrm{d}\ell}{\mathrm{d}\phi},
\qquad
\varphi(0)=0,
\]

and advances by \(2\pi/n_{\mathrm{fp}}\) per field period. Periodic
differentiation uses Fourier collocation matrices on the same endpoint-free
grid.

## Topology and validity

`frame_helicity` counts the signed winding of the Frenet normal around the
axis over one field period. It is intrinsic to the axis. The first-order solve
later combines it with the flux and covariant-field signs.

Frenet coordinates require nonzero speed and curvature everywhere.
`GeometryDiagnostics` reports:

- minimum speed and curvature;
- minimum cylindrical radius;
- maximum frame orthogonality error;
- minimum frame determinant;
- Boolean Frenet and cylindrical-coordinate validity.

Invalid samples are represented by non-finite frame/torsion values and a false
validity flag. High-level solvers reject them with a structured report rather
than allowing them to propagate silently.
