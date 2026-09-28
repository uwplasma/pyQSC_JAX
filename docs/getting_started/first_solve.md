# First solve

The first-order quasi-axisymmetric example of {cite:t}`landreman2019numerical` has a
three-period axis $R = 1 + 0.045\cos 3\phi$, $Z = -0.045\sin 3\phi$ and $\bar\eta = -0.9$.

```python
import jax
jax.config.update("jax_enable_x64", True)
import pyqsc_jax as qsc

solution = qsc.Qsc(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3, etabar=-0.9, nphi=31)

print(solution.iota)                      # 0.41830690943387
print(solution.axis_length)               # 6.340238817434 m
print(solution.root_report.converged)     # True
print(solution.root_report.residual_norm) # sigma-equation residual
```

The run gives $\iota =$ {{ qa_iota }} and axis length {{ qa_axis_length }} m after
{{ qa_newton_iterations }} Newton iterations, with residual {{ qa_sigma_residual }}.

`Qsc` is a pyQSC-style shortcut for {func}`pyqsc_jax.solve`, which takes an explicit
{class}`pyqsc_jax.Axis`:

```python
axis = qsc.Axis(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3)
solution = qsc.solve(axis=axis, etabar=-0.9, nphi=31)
```

## Second order, diagnostics, plot

```python
solution = qsc.Qsc(rc=[1.0, 0.155, 0.0102], zs=[0.0, 0.154, 0.0111], nfp=2,
                   etabar=0.64, B2c=-0.00322, order="r2", nphi=61)

solution.linear_report.converged   # second-order linear solve
solution.B20_variation             # max(B20) - min(B20)  [T/m^2]
solution.r_singularity             # first singularity of the near-axis map [m]
solution.DMerc_times_r2            # Mercier criterion (zero here: vacuum)
solution.grad_grad_B_axis.shape    # (61, 3, 3, 3)

from pyqsc_jax.plotting import plot_surface_3d
figure, ax = plot_surface_3d(solution, radius=0.1)
```

Always check `root_report.converged` (and `linear_report.converged` at second order)
before using a result. Continue with {doc}`../user_guide/solutions` and the
{doc}`../examples/index`.
