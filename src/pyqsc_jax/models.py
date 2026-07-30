"""Small immutable result and diagnostic models."""

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GeometryDiagnostics:
    """Validity and conditioning information for sampled axis geometry."""

    minimum_speed: jax.Array
    minimum_curvature: jax.Array
    minimum_cylindrical_radius: jax.Array
    maximum_frame_orthogonality_error: jax.Array
    minimum_frame_determinant: jax.Array
    frenet_valid: jax.Array
    cylindrical_coordinates_valid: jax.Array
