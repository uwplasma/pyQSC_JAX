# Coordinates and conventions

The magnetic axis uses cylindrical azimuth \(\phi\). Boozer toroidal angle
\(\varphi\) is normalized to advance by \(2\pi/n_{\mathrm{fp}}\) per field
period. The helical angle is

\[
\vartheta=\theta-Nn_{\mathrm{fp}}\varphi,
\qquad
\iota_N=\iota+Nn_{\mathrm{fp}}.
\]

Canonical Cartesian vectors are ordered `(x, y, z)` and sampled arrays place
the toroidal index first. Cylindrical vectors are ordered `(R, phi, Z)`.
The Frenet basis is right-handed `(tangent, normal, binormal)`, while pyQSC's
historical Hessian representation uses derivative axes in
`(normal, binormal, tangent)` order.

`sG` is the sign of \(G_0\), `spsi` the sign of toroidal flux, and
`chi = sG * spsi` in current conversions. Torsion follows Landreman and
Sengupta, opposite to the original Garren--Boozer sign convention.

See the [full coordinate derivation](../theory/coordinates-and-conventions.md)
for Fourier definitions, frame topology, and validity diagnostics.
