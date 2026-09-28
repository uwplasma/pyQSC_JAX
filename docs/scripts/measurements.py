"""Numbers quoted in the theory, user-guide and development pages, and the
convergence and overview figures.

Writes ``convergence.png`` and ``overview.png`` and merges every quoted number
into ``measurements.json``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from common import COLORS, record, savefig, sci

import pyqsc_jax as qsc
from pyqsc_jax import plasma
from pyqsc_jax.plotting import surface_coordinates
from pyqsc_jax.solvers import RootSolveOptions

M: dict[str, object] = {}

# Landreman & Sengupta (2019), section 5.1 (vacuum QA, nfp = 2).
LS51 = dict(rc=[1.0, 0.155, 0.0102], zs=[0.0, 0.154, 0.0111], nfp=2, etabar=0.64, B2c=-0.00322)
# Landreman, Sengupta & Plunk (2019) first-order QA and QH examples.
QA = dict(rc=[1.0, 0.045], zs=[0.0, -0.045], nfp=3, etabar=-0.9)
QH = dict(rc=[1.0, 0.265], zs=[0.0, -0.21], nfp=4, etabar=-0.9)
# Stellarator database configuration 52521 (Curvo et al. 2025): finite pressure, I2 = 0.
DB52521 = dict(
    rc=[1.0, -0.5415884, 0.029195854, 0.0048646266],
    zs=[0.0, -0.57113713, 0.029922731, 0.0041398546],
    nfp=4,
    etabar=1.1396117,
    B2c=-0.050057083,
    p2=-28248.188,
)


def first_order_cases() -> None:
    qa = qsc.Qsc(**QA, nphi=31)
    qh = qsc.Qsc(**QH, nphi=31)
    M["qa_iota"] = f"{float(qa.iota):.14f}"
    M["qa_axis_length"] = f"{float(qa.axis_length):.12f}"
    M["qa_sigma_residual"] = sci(qa.root_report.residual_norm)
    M["qa_newton_iterations"] = int(qa.root_report.iterations)
    M["qh_iota"] = f"{float(qh.iota):.10f}"
    M["qh_iotaN"] = f"{float(qh.iotaN):.10f}"
    M["qh_helicity"] = int(qh.helicity)


def second_and_third_order() -> None:
    r2 = qsc.Qsc(**LS51, nphi=61, order="r2")
    residuals = qsc.second_order_residuals(r2)
    M["ls51_iota"] = f"{float(r2.iota):.10f}"
    M["ls51_B20_mean"] = f"{float(r2.B20_mean):.8f}"
    M["ls51_B20_variation"] = f"{float(r2.B20_variation):.6f}"
    M["ls51_r2_residual"] = sci(residuals.maximum_absolute)
    M["ls51_linear_condition"] = f"{float(r2.linear_report.condition_estimate):.3g}"
    M["ls51_r_singularity"] = f"{float(r2.r_singularity):.6f}"
    M["ls51_singularity_residual"] = sci(r2.singularity.maximum_residual_norm if r2.singularity else qsc.singularity_diagnostics(r2).maximum_residual_norm)
    jet = qsc.total_field_jet(r2)
    M["ls51_jet_divergence"] = sci(jet.maximum_divergence)
    M["ls51_jet_asymmetry"] = sci(jet.maximum_derivative_asymmetry)
    M["ls51_jet_gradient_error"] = sci(jet.maximum_gradient_error)
    M["ls51_jet_divergence_gradient"] = sci(jet.maximum_divergence_gradient)
    M["ls51_L_grad_grad_B"] = f"{float(jet.L_grad_grad_B.min()):.4f}"
    for nphi in (61, 121):
        r3 = qsc.Qsc(**LS51, nphi=nphi, order="r3")
        M[f"ls51_r3_flux_residual_{nphi}"] = sci(r3.flux_constraint_residual)
        M[f"ls51_r3_consistency_{nphi}"] = sci(r3.consistency_error)


def plasma_split() -> None:
    solution = qsc.Qsc(**DB52521, nphi=61, order="r2")
    result = qsc.plasma_hessian_on_axis(solution, formal_radius=0.15)
    gradient = result.field
    field = gradient.field
    M["db_formal_radius"] = 0.15
    M["db_r_singularity"] = f"{float(solution.r_singularity):.4f}"
    M["db_matching_scale_error"] = sci(field.maximum_matching_scale_error)
    M["db_field_remainder"] = sci(field.estimated_field_remainder)
    M["db_hessian_remainder"] = sci(result.estimated_hessian_remainder)
    M["db_plasma_fraction_mean"] = f"{float(jnp.mean(jnp.linalg.norm(field.field, axis=-1))):.3e}"
    M["db_external_gradient_asymmetry"] = sci(gradient.maximum_external_asymmetry)
    M["db_external_gradient_trace"] = sci(gradient.maximum_external_trace)
    M["db_external_hessian_symmetry"] = sci(result.maximum_external_symmetry_error)
    M["db_external_hessian_trace"] = sci(result.maximum_external_trace)
    M["db_DMerc"] = f"{float(solution.DMerc_times_r2):.5f}"
    M["db_DWell"] = f"{float(solution.DWell_times_r2):.5f}"
    M["db_iota"] = f"{float(solution.iota):.6f}"
    M["db_plasma_divergence"] = sci(gradient.maximum_divergence)

    # Finite-current channel: Ampere's law for the leading plasma gradient.
    current = qsc.Qsc(**{**DB52521, "p2": 0.0}, I2=0.3, nphi=61, order="r2")
    current_gradient = qsc.plasma_gradient_on_axis(current, formal_radius=0.15)
    M["current_ampere_error"] = sci(current_gradient.maximum_ampere_error)
    M["current_external_gradient_asymmetry"] = sci(current_gradient.maximum_external_asymmetry)

    # Size of the order switch at I2 = 0 (see plasma_gradient_on_axis).
    near_zero = qsc.Qsc(**DB52521, I2=1e-9, nphi=61, order="r2")
    jump = qsc.plasma_gradient_on_axis(near_zero, formal_radius=0.15).gradient - gradient.gradient
    M["zero_current_jump"] = sci(jnp.max(jnp.abs(jump)))
    M["zero_current_gradient_norm"] = sci(jnp.max(jnp.abs(gradient.gradient)))


def _uncorrected_axis_integral(solution) -> jax.Array:
    """Punctured trapezoidal rule without the kink correction (reference only)."""

    source_varphi, position, tangent, weights = plasma._full_torus_axis_samples(solution)
    nphi = solution.inputs.nphi
    geometry = solution.geometry
    integrand = plasma.paired_axis_integrand(
        geometry.position_cartesian,
        geometry.curvature,
        geometry.binormal_cartesian,
        source_varphi[None, :] - source_varphi[:nphi, None],
        position[None, :, :],
        tangent[None, :, :],
        geometry.abs_G0_over_B0,
    )
    return jnp.sum(weights[None, :, None] * integrand, axis=1)


def convergence() -> None:
    grid = [15, 21, 31, 41, 61, 81, 121]
    reference_nphi = 361
    ref_r2 = qsc.Qsc(**LS51, nphi=reference_nphi, order="r2")
    ref_db = qsc.Qsc(**DB52521, nphi=reference_nphi, order="r2")
    ref_integral = qsc.regularized_axis_integral(ref_db)[0]
    iota_error, B20_error, divergence, integral_error, integral_plain = [], [], [], [], []
    for nphi in grid:
        r2 = qsc.Qsc(**LS51, nphi=nphi, order="r2")
        iota_error.append(abs(float(r2.iota - ref_r2.iota)))
        scale = float(r2.R0[0]) ** 2 / float(r2.inputs.B0)
        B20_error.append(abs(float(r2.B20[0] - ref_r2.B20[0])) * scale)
        divergence.append(float(qsc.total_field_jet(r2).maximum_divergence) * float(r2.R0[0]) / float(r2.inputs.B0))
        db = qsc.Qsc(**DB52521, nphi=nphi, order="r2")
        integral_error.append(float(jnp.linalg.norm(qsc.regularized_axis_integral(db)[0] - ref_integral)))
        integral_plain.append(float(jnp.linalg.norm(_uncorrected_axis_integral(db)[0] - ref_integral)))
    floor = 1e-16
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    axes[0].semilogy(grid, np.maximum(iota_error, floor), "o-", label=r"$|\iota-\iota_{ref}|$")
    axes[0].semilogy(grid, np.maximum(B20_error, floor), "s-", label=r"$|B_{20}-B_{20,ref}|\,R_0^2/B_0$")
    axes[0].set_title("(a) spectral collocation, LS 5.1")
    axes[1].semilogy(grid, divergence, "o-", color=COLORS[2])
    axes[1].set_title(r"(b) field jet: $R_0\max|\nabla\cdot\mathbf{B}|/B_0$")
    axes[2].loglog(grid, integral_plain, "s--", color=COLORS[3], label="punctured trapezoid")
    axes[2].loglog(grid, integral_error, "o-", color=COLORS[1], label="with kink correction")
    h = np.asarray(grid, dtype=float)
    axes[2].loglog(h, integral_plain[0] * (h / h[0]) ** -2, ":", color="grey", label=r"$h^2$")
    axes[2].loglog(h, integral_error[0] * (h / h[0]) ** -4, "-.", color="grey", label=r"$h^4$")
    axes[2].set_xticks(grid)
    axes[2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_title(r"(c) axis integral $\mathcal{R}(0)$, DB 52521")
    for axis in axes:
        axis.set_xlabel(r"$n_\phi$ per field period")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    savefig(figure, "convergence")
    M["convergence_reference_nphi"] = reference_nphi
    M["convergence_iota_error_61"] = sci(iota_error[grid.index(61)])
    M["convergence_B20_error_61"] = sci(B20_error[grid.index(61)])
    M["convergence_integral_error_61"] = sci(integral_error[grid.index(61)])
    M["convergence_integral_plain_61"] = sci(integral_plain[grid.index(61)])


def differentiation() -> None:
    axis = qsc.Axis(rc=LS51["rc"], zs=LS51["zs"], nfp=2)

    def variation(etabar):
        return qsc.solve(axis=axis, etabar=etabar, B2c=LS51["B2c"], nphi=61, order=2).B20_variation

    etabar = jnp.asarray(LS51["etabar"])
    _, tangent = jax.jvp(variation, (etabar,), (jnp.ones(()),))
    step = 1e-5
    fd = (variation(etabar + step) - variation(etabar - step)) / (2 * step)
    M["ad_B20_variation_relative_error"] = sci(abs(tangent - fd) / abs(fd))

    def r_singularity(etabar):
        return qsc.solve(axis=axis, etabar=etabar, B2c=LS51["B2c"], nphi=61, order=2).r_singularity

    _, tangent = jax.jvp(r_singularity, (etabar,), (jnp.ones(()),))
    fd = (r_singularity(etabar + step) - r_singularity(etabar - step)) / (2 * step)
    M["ad_r_singularity_relative_error"] = sci(abs(tangent - fd) / abs(fd))

    # A deliberately truncated Newton solve returns NaN tangents, never a wrong derivative.
    truncated = RootSolveOptions(max_steps=1)

    def truncated_iota(etabar):
        return qsc.solve(axis=axis, etabar=etabar, nphi=61, root_options=truncated).iota

    failed = qsc.solve(axis=axis, etabar=etabar, nphi=61, root_options=truncated)
    _, tangent = jax.jvp(truncated_iota, (etabar,), (jnp.ones(()),))
    M["truncated_converged"] = str(bool(failed.root_report.converged))
    M["truncated_tangent"] = str(float(tangent))

    # Timings (machine-specific; median of repeated warm calls).
    def timed(function, repeats=20):
        start = time.perf_counter()
        jax.block_until_ready(function())
        compile_time = time.perf_counter() - start
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(function())
            samples.append(time.perf_counter() - start)
        return compile_time, float(np.median(samples))

    jax.clear_caches()
    compile_r2, warm_r2 = timed(lambda: qsc.solve(axis=axis, etabar=0.64, B2c=-0.00322, nphi=61, order=2))
    _, warm_r2_new = timed(lambda: qsc.solve(axis=axis, etabar=0.65, B2c=-0.00322, nphi=61, order=2))
    grad = jax.jit(jax.grad(variation))
    compile_grad, warm_grad = timed(lambda: grad(etabar))
    M["time_r2_compile_s"] = f"{compile_r2:.1f}"
    M["time_r2_warm_ms"] = f"{1e3 * warm_r2:.2g}"
    M["time_r2_newvalue_ms"] = f"{1e3 * warm_r2_new:.2g}"
    M["time_grad_compile_s"] = f"{compile_grad:.1f}"
    M["time_grad_warm_ms"] = f"{1e3 * warm_grad:.2g}"
    M["time_machine"] = f"{jax.devices()[0].platform.upper()}, {__import__('platform').machine()}"


def overview() -> None:
    solution = qsc.Qsc(**DB52521, nphi=121, order="r2")
    radius = 0.12
    x, y, z = (np.asarray(value) for value in surface_coordinates(solution, radius=radius, ntheta=48))
    figure = plt.figure(figsize=(7.0, 4.2))
    axis = figure.add_subplot(111, projection="3d")
    from matplotlib.colors import LightSource

    light = LightSource(azdeg=315, altdeg=45)
    colors = light.shade(z, cmap=plt.get_cmap("Blues_r"), vert_exag=0.3, blend_mode="soft")
    axis.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, rstride=1, cstride=1)
    phi = np.linspace(0, 2 * np.pi, 400)
    samples = qsc.geometry.evaluate_axis(solution.inputs.axis, jnp.asarray(phi))
    axis.plot(np.asarray(samples.R) * np.cos(phi), np.asarray(samples.R) * np.sin(phi), np.asarray(samples.Z), color="k", linewidth=1.0)
    span = max(np.ptp(x), np.ptp(y)) / 2
    axis.set_xlim(-span, span)
    axis.set_ylim(-span, span)
    axis.set_zlim(-span / 2.5, span / 2.5)
    axis.set_box_aspect((1, 1, 0.4))
    axis.set_axis_off()
    axis.view_init(elev=32, azim=-60)
    figure.subplots_adjust(0, 0, 1, 1)
    savefig(figure, "overview", dpi=130)
    M["overview_radius"] = radius


if __name__ == "__main__":
    first_order_cases()
    second_and_third_order()
    plasma_split()
    differentiation()
    convergence()
    overview()
    for key, value in sorted(M.items()):
        print(f"{key:40s} {value}")
    record("measurements.py", M)
