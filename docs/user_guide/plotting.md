# Plotting

`pyqsc_jax.plotting` needs Matplotlib (the `plot` extra). It imports Matplotlib only when a
function is called and never changes global Matplotlib settings. Every function returns
`(figure, axes)` and accepts an existing `ax` to draw into.

| function | draws |
|---|---|
| `plot_axis(solution_or_axis)` | the full-torus axis in 3-D |
| `surface_coordinates(solution, radius, ntheta)` | full-torus Cartesian surface arrays (no plotting) |
| `surface_field_strength(solution, radius, ntheta)` | near-axis \|B\| on the same grid: $B_0(1 + r\bar\eta\cos\vartheta)$, plus $r^2(B_{20} + B_{2c}\cos 2\vartheta + B_{2s}\sin 2\vartheta)$ for r2/r3 (no $r^3$ term) |
| `surface_normal_field_error(solution, field, radius, ntheta)` | $B\cdot n/|B|$ of an external field on the same grid, with the outward normal of the sampled surface |
| `plot_surface_3d(solution, radius, ntheta, color_by="B", field=None)` | the surface at `radius` and the axis, with equal x, y, z scales, coloured by near-axis \|B\| (default), by height (`"height"`) or by $\log_{10}(|B\cdot n|/|B|)$ of `field` (`"Bn"`) |
| `plot_b20(solution)` | $B_{20} - \langle B_{20}\rangle$ against Boozer angle |
| `plot_field_split_components(result, solution)` | total, plasma and external field in the Frenet frame |
| `plot_field_jet_norms(result)` | Frobenius norms of the three field jets |
| `field_split_frenet_components(result, solution)` | the arrays behind the split plot |

```python
from pyqsc_jax.plotting import plot_b20, plot_surface_3d

figure, ax = plot_b20(solution, label="r2")
figure, ax = plot_surface_3d(solution, radius=0.05, ntheta=48)
figure.savefig("surface.png", dpi=150)

# Error field of coils (or any field): a callable B(points) with points of shape (n, 3),
# e.g. jax.vmap(biot_savart.B), or an array of shape (ntheta, nphi_plot, 3).
figure, ax = plot_surface_3d(solution, radius=0.05, color_by="Bn", field=coil_field)
```

`color_by="Bn"` uses a logarithmic viridis scale, so a vanishing normal field is dark, not
white; `field` is required and a missing one raises `ValueError`.

Choose the plotting radius below `solution.r_singularity`; beyond it the truncated surface
can self-intersect. The surface includes every solved order (the $r^3$ correction when
`order="r3"`). For a boundary on uniform cylindrical angles use
{func}`~pyqsc_jax.vmec_boundary` instead ({doc}`../theory/vmec`).

The ESSOS adapter keeps pyQSC-style `plot` and `get_boundary` methods; see
{doc}`../examples/08_boundary_and_coordinates`.
