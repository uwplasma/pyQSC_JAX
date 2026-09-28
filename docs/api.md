# API reference

The public API is the set of names exported by `pyqsc_jax` (its `__all__`). Everything
else is internal and may change. Units and scope are stated in each docstring; all
functions need 64-bit JAX.

## Solving

```{eval-rst}
.. autofunction:: pyqsc_jax.solve
.. autofunction:: pyqsc_jax.Qsc
.. autoclass:: pyqsc_jax.Axis
   :members: stellarator_symmetric, from_dofs, dofs, nfourier, with_dofs, stellarator_symmetry_residual
.. autoclass:: pyqsc_jax.RootSolveOptions
.. autofunction:: pyqsc_jax.solve_third_order
.. autofunction:: pyqsc_jax.second_order_residuals
```

## Results and reports

```{eval-rst}
.. autoclass:: pyqsc_jax.NearAxisSolution
   :members: with_diagnostics
   :undoc-members: false
.. autoclass:: pyqsc_jax.NearAxisInputs
   :no-members:
.. autoclass:: pyqsc_jax.SecondOrderData
   :no-members:
.. autoclass:: pyqsc_jax.ThirdOrderData
   :no-members:
.. autoclass:: pyqsc_jax.RootSolveReport
   :no-members:
.. autoclass:: pyqsc_jax.LinearSolveReport
   :no-members:
```

## Diagnostics

```{eval-rst}
.. autofunction:: pyqsc_jax.total_field_jet
.. autofunction:: pyqsc_jax.singularity_diagnostics
.. autofunction:: pyqsc_jax.mercier_diagnostics
.. autofunction:: pyqsc_jax.b20_diagnostics
.. autofunction:: pyqsc_jax.optimize_B2c
.. autofunction:: pyqsc_jax.optimal_B2c_value
.. autoclass:: pyqsc_jax.FieldJet
   :no-members:
.. autoclass:: pyqsc_jax.SingularityDiagnostics
   :no-members:
.. autoclass:: pyqsc_jax.MercierDiagnostics
   :no-members:
.. autoclass:: pyqsc_jax.B20Diagnostics
   :no-members:
.. autoclass:: pyqsc_jax.B2cOptimizationResult
   :no-members:
```

## Plasma field

```{eval-rst}
.. autofunction:: pyqsc_jax.plasma_current_source
.. autofunction:: pyqsc_jax.regularized_axis_integral
.. autofunction:: pyqsc_jax.plasma_field_on_axis
.. autofunction:: pyqsc_jax.plasma_gradient_on_axis
.. autofunction:: pyqsc_jax.plasma_hessian_on_axis
.. autoclass:: pyqsc_jax.PlasmaCurrentSource
   :no-members:
.. autoclass:: pyqsc_jax.PlasmaFieldData
   :no-members:
.. autoclass:: pyqsc_jax.PlasmaGradientData
   :no-members:
.. autoclass:: pyqsc_jax.PlasmaHessianData
   :no-members:
```

## VMEC and VMEX

```{eval-rst}
.. autofunction:: pyqsc_jax.to_vmec
.. autofunction:: pyqsc_jax.vmec_boundary
.. autofunction:: pyqsc_jax.uniform_cylindrical_surface
.. autoclass:: pyqsc_jax.VmecBoundary
   :no-members:
.. autoclass:: pyqsc_jax.VmecExport
   :no-members:
.. autoclass:: pyqsc_jax.VmecInputParameters
   :no-members:
.. autofunction:: pyqsc_jax.to_vmex_problem
.. autofunction:: pyqsc_jax.solve_vmex
.. autofunction:: pyqsc_jax.vmex_radial_quantities
.. autoclass:: pyqsc_jax.VmexProblem
   :members: finite_beta, parameters_for, quantities, solve
.. autoclass:: pyqsc_jax.VmexEquilibrium
   :no-members:
.. autoclass:: pyqsc_jax.VmexRadialQuantities
   :no-members:
```

## ESSOS adapter and plotting

These live in submodules and are not part of `__all__`.

```{eval-rst}
.. autoclass:: pyqsc_jax.near_axis.near_axis
   :members: dofs, x, get_boundary, to_vmec, plot, B_mag, to_vtk
.. automodule:: pyqsc_jax.plotting
   :members: plot_axis, surface_coordinates, plot_surface_3d, plot_b20, plot_field_split_components, plot_field_jet_norms, field_split_frenet_components
```
