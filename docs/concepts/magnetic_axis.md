# Magnetic axis

`Axis` stores general Fourier series

\[
R(\phi)=\sum_n[R_{cn}\cos(nn_{\mathrm{fp}}\phi)+R_{sn}\sin(nn_{\mathrm{fp}}\phi)],
\]

\[
Z(\phi)=\sum_n[Z_{cn}\cos(nn_{\mathrm{fp}}\phi)+Z_{sn}\sin(nn_{\mathrm{fp}}\phi)].
\]

The public coefficient order is always `rc`, `rs`, `zc`, `zs`. Arrays are
zero-padded to a common Fourier length. Stellarator symmetry is the special
case `rs = zc = 0`; it is not assumed by the geometry core.

`compute_axis_geometry` evaluates analytic Fourier derivatives, arc length,
curvature, torsion, Frenet frames, helicity, the \(\phi\leftrightarrow\varphi\)
map, and periodic spectral operators. Reject a result if
`geometry.diagnostics.frenet_valid` or
`cylindrical_coordinates_valid` is false. Curvature approaching zero is a
coordinate failure, not a benign numerical warning.

Implementation: `pyqsc_jax.axis` and `pyqsc_jax.geometry`. Validation:
`tests/physics/test_geometry.py` and the audited upstream regression data in
`tests/regression/test_geometry_reference.py`.
