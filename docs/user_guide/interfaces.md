# VMEX and ESSOS interfaces

## VMEC input files

```python
export = qsc.to_vmec(solution, "input.qa", r=0.005, mpol=12, ntor=14,
                     parameters={"ns_array": (15, 31), "ftol_array": (1e-10, 1e-12),
                                 "niter_array": (2000, 4000)})
export.boundary.maximum_R_reconstruction_error
```

`to_vmec` writes a fixed-boundary `&INDATA` file and returns a
{class}`~pyqsc_jax.VmecExport` with the path, the {class}`~pyqsc_jax.VmecBoundary`, the
profile values and `lasym`. It raises before writing if the boundary angle inversion
fails; reduce `r` or raise `newton_iterations`. See {doc}`../theory/vmec` for the
algorithm, the file contents and the `MPOL = mpol + 1` convention.
{func}`~pyqsc_jax.vmec_boundary` is the jittable, differentiable part without file I/O.

## VMEX: differentiable fixed-boundary equilibria

[VMEX](https://github.com/uwplasma/vmex) is a JAX implementation of VMEC with implicit
derivatives. It is optional (`pip install 'pyqsc-jax[vmex]'`) and imported only when one of
these functions is called.

```python
problem = qsc.to_vmex_problem(solution, r=0.02, qs_surfaces=(0.25, 0.5, 0.75, 1.0),
                              mpol=6, ntor=6, ns_array=(15, 31))
result = problem.solve()                  # VmexEquilibrium: raw VMEX solution + quantities
q = result.quantities                     # VmexRadialQuantities
q.iota, q.quasisymmetry, q.magnetic_well, q.aspect, q.volume

# gradient through VMEX's converged fixed point
well, grad = jax.value_and_grad(
    lambda p: qsc.vmex_radial_quantities(problem, p).magnetic_well)(problem.parameters)
```

{func}`~pyqsc_jax.to_vmex_problem` builds the boundary, pressure (`AM`), current (`CURTOR`)
and flux (`PHIEDGE`) exactly as `to_vmec` does and returns a reusable
{class}`~pyqsc_jax.VmexProblem`: static resolution and solver controls plus a
differentiable parameter pytree. `problem.parameters_for(new_solution)` maps another
near-axis solution with the same `nfp` into that pytree, so gradients flow from the
near-axis inputs through the boundary map and the VMEX equilibrium. The quasisymmetry
helicity defaults to $(M, N) = (1, -N_\mathrm{frame})$; the traceable quasisymmetry profile
requires stellarator symmetry (pass `qs_surfaces=()` for asymmetric boundaries).
`VmexRadialQuantities.iota` uses the pyQSC_JAX sign; `iota_vmec` keeps VMEC's.
The interface was validated against VMEX commit `2a40d756`
(`problem.validated_commit`); {doc}`../examples/14_vmex_radial_profiles` runs it.

## ESSOS

[ESSOS](https://github.com/uwplasma/ESSOS) uses pyQSC_JAX through two entry points.

**Near-axis field.** `pyqsc_jax.near_axis.near_axis` is a thin mutable adapter over
{func}`~pyqsc_jax.solve` for stellarator-symmetric axes. It keeps the pyQSC/ESSOS
interface: constructor arguments `rc, zs, etabar, B0, sigma0, I2, nphi, spsi, sG, nfp,
order, B2c, p2, B2s`; the degrees of freedom `dofs`/`x` ordered `rc, zs, etabar` (setting
them re-solves); the historical `(component, sample)` layouts of `B_axis`, `grad_B_axis`
and `grad_grad_B_axis`; and `get_boundary`, `to_vmec`, `plot`, `B_mag`, `to_vtk`. Every
other attribute is delegated to the immutable `solution`. It requires odd `nphi`. New code
should use `solve` directly.

**Coil targets.** For finite-beta coil design, ESSOS takes the external field, gradient and
Hessian from {func}`~pyqsc_jax.plasma_hessian_on_axis`:

```python
result = qsc.plasma_hessian_on_axis(solution, formal_radius=0.15)
B_ext = result.field.external_field                   # (nphi, 3)
dB_ext = result.field.external_gradient_independent   # (nphi, 5)  xx, yy, xy, xz, yz
ddB_ext = result.external_hessian_independent         # (nphi, 7)  xxx ... yyz
```

pyQSC_JAX never imports ESSOS. ESSOS imports `near_axis`, `Qsc`, `solve`, `Axis`, the
plasma functions and STF packers of `pyqsc_jax.plasma`, `to_vmec`, `vmec_boundary`,
`uniform_cylindrical_surface`, `geometry.compute_axis_geometry` and `second_order.MU0`;
these module paths are kept stable, and the adapter's attributes and methods are checked by
`tests/compatibility`.
