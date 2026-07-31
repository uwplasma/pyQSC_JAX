# VMEC export

```{eval-rst}
.. automodule:: pyqsc_jax.vmec
   :members:
   :undoc-members:
   :show-inheritance:
```

`to_vmec` is also available as a method on the legacy
`pyqsc_jax.near_axis.near_axis` adapter. That method preserves the upstream
call form and populates `RBC`, `RBS`, `ZBC`, and `ZBS` compatibility arrays.

`VmecBoundary.toroidal_angle_converged` and
`maximum_toroidal_angle_residual` diagnose the vectorized inversion. The
writer never emits an input when the configured tolerance is not reached.
