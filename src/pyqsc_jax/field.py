"""Surface-free total magnetic field jets on the magnetic axis."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pyqsc_jax.geometry import cylindrical_vector_to_cartesian
from pyqsc_jax.models import FieldJet, NearAxisSolution


def _differentiate_cylindrical_vector(
    vector: jax.Array,
    solution: NearAxisSolution,
) -> jax.Array:
    """Differentiate a vector with respect to Boozer ``varphi``.

    Cylindrical components are periodic over one field period, while fixed
    Cartesian components generally are not. The two connection terms below
    account for rotation of the cylindrical basis.
    """

    derivative = solution.geometry.d_d_varphi @ vector
    d_phi_d_varphi = 1 / solution.geometry.d_varphi_d_phi
    derivative = derivative.at[:, 0].add(-d_phi_d_varphi * vector[:, 1])
    return derivative.at[:, 1].add(d_phi_d_varphi * vector[:, 0])


def _to_cartesian(vector: jax.Array, solution: NearAxisSolution) -> jax.Array:
    return cylindrical_vector_to_cartesian(vector, solution.phi)


def total_field_jet(solution: NearAxisSolution) -> FieldJet:
    """Compute ``B``, ``grad(B)``, and ``grad(grad(B))`` on the axis.

    This is the regular-coordinate chain rule in equations (55)--(83) of the
    surface-free plasma/coil derivation. It requires the complete second-order
    near-axis solution but no finite-radius surface.
    """

    r2 = solution.second_order
    if r2 is None:
        raise ValueError("A second-order solution is required for the field Hessian.")

    geometry = solution.geometry
    inputs = solution.inputs
    length_scale = geometry.abs_G0_over_B0
    tangent = geometry.tangent_cylindrical
    normal = geometry.normal_cylindrical
    binormal = geometry.binormal_cylindrical

    d1 = solution.X1c[:, None] * normal + solution.Y1c[:, None] * binormal
    d2 = solution.Y1s[:, None] * binormal
    h11 = (
        2 * (r2.X20 + r2.X2c)[:, None] * normal
        + 2 * (r2.Y20 + r2.Y2c)[:, None] * binormal
        + 2 * (r2.Z20 + r2.Z2c)[:, None] * tangent
    )
    h12 = (
        2 * r2.X2s[:, None] * normal
        + 2 * r2.Y2s[:, None] * binormal
        + 2 * r2.Z2s[:, None] * tangent
    )
    h22 = (
        2 * (r2.X20 - r2.X2c)[:, None] * normal
        + 2 * (r2.Y20 - r2.Y2c)[:, None] * binormal
        + 2 * (r2.Z20 - r2.Z2c)[:, None] * tangent
    )

    derivative = lambda vector: _differentiate_cylindrical_vector(vector, solution)  # noqa: E731
    V0 = length_scale * tangent
    V1 = derivative(d1) + solution.iotaN * d2
    V2 = derivative(d2) - solution.iotaN * d1
    V11 = derivative(h11) + 2 * solution.iotaN * h12
    V12 = derivative(h12) + solution.iotaN * (h22 - h11)
    V22 = derivative(h22) - 2 * solution.iotaN * h12

    p0 = inputs.B0**2 / solution.G0
    p1 = 2 * inputs.B0**2 * inputs.etabar / solution.G0
    flux_term = inputs.B0**2 * (r2.G2 + solution.iota * inputs.I2) / solution.G0**2
    C11 = (
        inputs.B0**2 * inputs.etabar**2 + 2 * inputs.B0 * (r2.B20 + inputs.B2c)
    ) / solution.G0 - flux_term
    C22 = 2 * inputs.B0 * (r2.B20 - inputs.B2c) / solution.G0 - flux_term

    field_coordinate_gradient_cylindrical = jnp.stack(
        (
            p0 * derivative(V0),
            p1 * V0 + p0 * V1,
            p0 * V2,
        ),
        axis=-1,
    )
    field_coordinate_hessian_cylindrical = jnp.zeros(
        (inputs.nphi, 3, 3, 3),
        dtype=field_coordinate_gradient_cylindrical.dtype,
    )
    coordinate_hessian_values = {
        (0, 0): p0 * derivative(derivative(V0)),
        (0, 1): p1 * derivative(V0) + p0 * derivative(V1),
        (0, 2): p0 * derivative(V2),
        (1, 1): 2 * C11[:, None] * V0 + 2 * p1 * V1 + p0 * V11,
        (1, 2): p1 * V2 + p0 * V12,
        (2, 2): 2 * C22[:, None] * V0 + p0 * V22,
    }
    for (first, second), value in coordinate_hessian_values.items():
        field_coordinate_hessian_cylindrical = field_coordinate_hessian_cylindrical.at[
            :, :, first, second
        ].set(value)
        field_coordinate_hessian_cylindrical = field_coordinate_hessian_cylindrical.at[
            :, :, second, first
        ].set(value)

    field_coordinate_gradient = jnp.stack(
        [
            _to_cartesian(field_coordinate_gradient_cylindrical[:, :, index], solution)
            for index in range(3)
        ],
        axis=-1,
    )
    field_coordinate_hessian = jnp.stack(
        [
            jnp.stack(
                [
                    _to_cartesian(
                        field_coordinate_hessian_cylindrical[:, :, first, second],
                        solution,
                    )
                    for second in range(3)
                ],
                axis=-1,
            )
            for first in range(3)
        ],
        axis=-1,
    )

    coordinate_jacobian = jnp.stack(
        (
            length_scale * geometry.tangent_cartesian,
            _to_cartesian(d1, solution),
            _to_cartesian(d2, solution),
        ),
        axis=-1,
    )
    coordinate_hessian = jnp.zeros_like(field_coordinate_hessian)
    map_hessian_values = {
        (0, 0): length_scale**2 * solution.curvature[:, None] * geometry.normal_cartesian,
        (0, 1): _to_cartesian(derivative(d1), solution),
        (0, 2): _to_cartesian(derivative(d2), solution),
        (1, 1): _to_cartesian(h11, solution),
        (1, 2): _to_cartesian(h12, solution),
        (2, 2): _to_cartesian(h22, solution),
    }
    for (first, second), value in map_hessian_values.items():
        coordinate_hessian = coordinate_hessian.at[:, :, first, second].set(value)
        coordinate_hessian = coordinate_hessian.at[:, :, second, first].set(value)

    inverse_coordinate_jacobian = jnp.linalg.inv(coordinate_jacobian)
    gradient = jnp.einsum(
        "nia,naj->nij",
        field_coordinate_gradient,
        inverse_coordinate_jacobian,
    )
    hessian = jnp.einsum(
        "niab,naj,nbk->nijk",
        field_coordinate_hessian,
        inverse_coordinate_jacobian,
        inverse_coordinate_jacobian,
    ) - jnp.einsum(
        "nia,nal,nlcb,ncj,nbk->nijk",
        field_coordinate_gradient,
        inverse_coordinate_jacobian,
        coordinate_hessian,
        inverse_coordinate_jacobian,
        inverse_coordinate_jacobian,
    )
    field = p0 * _to_cartesian(V0, solution)

    frenet_basis = jnp.stack(
        (
            geometry.normal_cartesian,
            geometry.binormal_cartesian,
            geometry.tangent_cartesian,
        ),
        axis=-1,
    )
    field_first_frenet = jnp.einsum(
        "nia,nijk,njb,nkc->nabc",
        frenet_basis,
        hessian,
        frenet_basis,
        frenet_basis,
    )
    hessian_frenet = jnp.transpose(field_first_frenet, (0, 2, 3, 1))

    norm_squared = jnp.sum(hessian**2, axis=(1, 2, 3))
    inverse_scale_vs_varphi = jnp.sqrt(jnp.sqrt(norm_squared) / (4 * inputs.B0))
    coordinate_determinant = jnp.linalg.det(coordinate_jacobian)
    divergence = jnp.trace(gradient, axis1=1, axis2=2)
    divergence_gradient = jnp.einsum("niik->nk", hessian)
    return FieldJet(
        field=field,
        gradient=gradient,
        hessian=hessian,
        hessian_frenet=hessian_frenet,
        coordinate_jacobian=coordinate_jacobian,
        inverse_coordinate_jacobian=inverse_coordinate_jacobian,
        coordinate_hessian=coordinate_hessian,
        minimum_absolute_coordinate_jacobian=jnp.min(jnp.abs(coordinate_determinant)),
        maximum_field_error=jnp.max(jnp.abs(field - solution.B_axis)),
        maximum_gradient_error=jnp.max(jnp.abs(gradient - solution.grad_B_axis)),
        maximum_divergence=jnp.max(jnp.abs(divergence)),
        maximum_derivative_asymmetry=jnp.max(jnp.abs(hessian - jnp.swapaxes(hessian, 2, 3))),
        maximum_divergence_gradient=jnp.max(jnp.abs(divergence_gradient)),
        grad_grad_B_inverse_scale_length_vs_varphi=inverse_scale_vs_varphi,
        L_grad_grad_B=1 / inverse_scale_vs_varphi,
        grad_grad_B_inverse_scale_length=jnp.max(inverse_scale_vs_varphi),
    )
