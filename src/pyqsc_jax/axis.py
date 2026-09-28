"""Fourier representations of the magnetic axis."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

ArrayLike = Any


def _as_coefficient_array(values: ArrayLike, dtype: jnp.dtype) -> jax.Array:
    array = jnp.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("Axis Fourier coefficients must be one-dimensional.")
    return array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Axis:
    """Immutable cylindrical Fourier representation of a closed magnetic axis.

    The convention over one field period is

    .. math::

        R(\\phi) = \\sum_n R_{cn}\\cos(n n_{fp}\\phi)
                  + R_{sn}\\sin(n n_{fp}\\phi),

        Z(\\phi) = \\sum_n Z_{cn}\\cos(n n_{fp}\\phi)
                  + Z_{sn}\\sin(n n_{fp}\\phi).

    All four coefficient arrays are padded to the same length. ``nfp`` is
    static pytree metadata, while the coefficient arrays are differentiable
    pytree leaves.
    """

    rc: jax.Array
    zs: jax.Array
    nfp: int = field(default=1, metadata={"static": True})
    rs: jax.Array = ()
    zc: jax.Array = ()

    coefficient_order: ClassVar[tuple[str, ...]] = ("rc", "rs", "zc", "zs")

    def __post_init__(self) -> None:
        if not isinstance(self.nfp, int) or isinstance(self.nfp, bool) or self.nfp < 1:
            raise ValueError("nfp must be a positive integer.")

        raw = tuple(jnp.asarray(values) for values in (self.rc, self.rs, self.zc, self.zs))
        dtype = jnp.result_type(jnp.asarray(0.0), *(array.dtype for array in raw))
        arrays = tuple(_as_coefficient_array(values, dtype) for values in raw)
        nfourier = max(array.size for array in arrays)
        if nfourier < 1:
            raise ValueError("At least one axis Fourier coefficient is required.")

        padded = tuple(jnp.pad(array, (0, nfourier - array.size)) for array in arrays)
        object.__setattr__(self, "rc", padded[0])
        object.__setattr__(self, "rs", padded[1])
        object.__setattr__(self, "zc", padded[2])
        object.__setattr__(self, "zs", padded[3])

    @classmethod
    def stellarator_symmetric(cls, *, rc: ArrayLike, zs: ArrayLike, nfp: int = 1) -> "Axis":
        """Construct an axis for which ``R`` is even and ``Z`` is odd."""

        return cls(rc=rc, zs=zs, nfp=nfp)

    @classmethod
    def from_dofs(cls, dofs: ArrayLike, *, nfp: int) -> "Axis":
        """Construct from four equal blocks ordered as ``rc, rs, zc, zs``."""

        array = jnp.asarray(dofs)
        if array.ndim != 1 or array.size % 4:
            raise ValueError(
                "Axis dofs must be a one-dimensional array with length divisible by 4."
            )
        nfourier = array.size // 4
        rc, rs, zc, zs = jnp.split(array, (nfourier, 2 * nfourier, 3 * nfourier))
        return cls(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp)

    @property
    def nfourier(self) -> int:
        """Number of retained Fourier modes, including mode zero."""

        return self.rc.size

    @property
    def dofs(self) -> jax.Array:
        """Coefficient vector ordered as ``rc, rs, zc, zs``."""

        return jnp.concatenate((self.rc, self.rs, self.zc, self.zs))

    @property
    def stellarator_symmetry_residual(self) -> jax.Array:
        """Largest coefficient forbidden by stellarator symmetry."""

        return jnp.maximum(jnp.max(jnp.abs(self.rs)), jnp.max(jnp.abs(self.zc)))

    def with_dofs(self, dofs: ArrayLike) -> "Axis":
        """Return a new axis with packed coefficients and the same ``nfp``."""

        return type(self).from_dofs(dofs, nfp=self.nfp)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AxisSamples:
    """Axis coordinates and their first three cylindrical-angle derivatives."""

    phi: jax.Array
    R: jax.Array
    Z: jax.Array
    d_R_d_phi: jax.Array
    d_Z_d_phi: jax.Array
    d2_R_d_phi2: jax.Array
    d2_Z_d_phi2: jax.Array
    d3_R_d_phi3: jax.Array
    d3_Z_d_phi3: jax.Array


def evaluate_axis(axis: Axis, phi: ArrayLike) -> AxisSamples:
    """Evaluate an axis and analytic derivatives at arbitrary toroidal angles."""

    phi = jnp.asarray(phi)
    mode = jnp.arange(axis.nfourier, dtype=phi.dtype) * axis.nfp
    angle = phi[..., None] * mode
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)

    R = jnp.sum(axis.rc * cosine + axis.rs * sine, axis=-1)
    Z = jnp.sum(axis.zc * cosine + axis.zs * sine, axis=-1)
    d_R = jnp.sum(mode * (-axis.rc * sine + axis.rs * cosine), axis=-1)
    d_Z = jnp.sum(mode * (-axis.zc * sine + axis.zs * cosine), axis=-1)
    mode2 = mode * mode
    d2_R = jnp.sum(-mode2 * (axis.rc * cosine + axis.rs * sine), axis=-1)
    d2_Z = jnp.sum(-mode2 * (axis.zc * cosine + axis.zs * sine), axis=-1)
    mode3 = mode2 * mode
    d3_R = jnp.sum(mode3 * (axis.rc * sine - axis.rs * cosine), axis=-1)
    d3_Z = jnp.sum(mode3 * (axis.zc * sine - axis.zs * cosine), axis=-1)

    return AxisSamples(
        phi=phi,
        R=R,
        Z=Z,
        d_R_d_phi=d_R,
        d_Z_d_phi=d_Z,
        d2_R_d_phi2=d2_R,
        d2_Z_d_phi2=d2_Z,
        d3_R_d_phi3=d3_R,
        d3_Z_d_phi3=d3_Z,
    )
