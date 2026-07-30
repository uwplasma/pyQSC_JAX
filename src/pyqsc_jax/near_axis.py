"""ESSOS-compatible mutable facade over the immutable near-axis core."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from pyqsc_jax.axis import Axis
from pyqsc_jax.first_order import solve
from pyqsc_jax.models import NearAxisSolution
from pyqsc_jax.solvers import implicit_dense_root

ArrayLike = Any


class near_axis:  # noqa: N801
    """Compatibility adapter for the historical ``near_axis`` interface.

    The canonical API is :func:`pyqsc_jax.solve`. This class intentionally
    keeps ESSOS's mutable ``x``/``dofs`` facade and historical component-axis
    ordering while delegating all near-axis physics to the immutable core.
    """

    def __init__(
        self,
        rc: ArrayLike = (1.0, 0.1),
        zs: ArrayLike = (0.0, 0.1),
        etabar: ArrayLike = 1.0,
        B0: ArrayLike = 1.0,
        sigma0: ArrayLike = 0.0,
        I2: ArrayLike = 0.0,
        nphi: int = 31,
        spsi: int = 1,
        sG: int = 1,
        nfp: int = 2,
        order: int | str = "r1",
        B2c: ArrayLike = 0.0,
        p2: ArrayLike = 0.0,
    ) -> None:
        if not isinstance(nphi, int) or isinstance(nphi, bool) or nphi < 3 or nphi % 2 == 0:
            raise ValueError("The compatibility API requires odd integer nphi >= 3.")

        self.rc = jnp.asarray(rc)
        self.zs = jnp.asarray(zs)
        if self.rc.ndim != 1 or self.zs.ndim != 1 or self.rc.size != self.zs.size:
            raise ValueError("rc and zs must be one-dimensional arrays of equal length.")
        self.etabar = jnp.asarray(etabar)
        self.B0 = jnp.asarray(B0)
        self.sigma0 = jnp.asarray(sigma0)
        self.I2 = jnp.asarray(I2)
        self.p2 = jnp.asarray(p2)
        self.B2c = jnp.asarray(B2c)
        self.nphi = nphi
        self.spsi = spsi
        self.sG = sG
        self.nfp = nfp
        self.order = order
        self.nfourier = self.rc.size
        self._dofs = jnp.concatenate((self.rc, self.zs, self.etabar[None]))
        self._refresh()

    def _canonical_solution(
        self,
        rc: ArrayLike,
        zs: ArrayLike,
        etabar: ArrayLike,
    ) -> NearAxisSolution:
        # Until r2/r3 milestones land, preserve the legacy behavior in which
        # those labels still returned the first-order subset.
        return solve(
            axis=Axis.stellarator_symmetric(rc=rc, zs=zs, nfp=self.nfp),
            etabar=etabar,
            B0=self.B0,
            sigma0=self.sigma0,
            I2=self.I2,
            p2=self.p2,
            B2c=self.B2c,
            nphi=self.nphi,
            order="r1",
            sG=self.sG,
            spsi=self.spsi,
        )

    @staticmethod
    def _legacy_tuple(solution: NearAxisSolution) -> tuple[jax.Array, ...]:
        geometry = solution.geometry
        normal = geometry.normal_cylindrical
        binormal = geometry.binormal_cylindrical
        return (
            solution.R0,
            solution.Z0,
            solution.sigma,
            solution.elongation,
            solution.B_axis.T,
            solution.grad_B_axis.T,
            solution.axis_length,
            solution.iota,
            solution.iotaN,
            solution.G0,
            solution.helicity,
            solution.X1c_untwisted,
            solution.X1s_untwisted,
            solution.Y1s_untwisted,
            solution.Y1c_untwisted,
            normal[:, 0],
            normal[:, 1],
            normal[:, 2],
            binormal[:, 0],
            binormal[:, 1],
            binormal[:, 2],
            solution.L_grad_B,
            1 / solution.L_grad_B,
            solution.torsion,
            solution.curvature,
            solution.varphi,
            geometry.samples.d_R_d_phi,
            geometry.samples.d_Z_d_phi,
        )

    def _refresh(self) -> None:
        solution = self._canonical_solution(self.rc, self.zs, self.etabar)
        self.solution = solution
        self.phi = solution.phi
        (
            self.R0,
            self.Z0,
            self.sigma,
            self.elongation,
            self.B_axis,
            self.grad_B_axis,
            self.axis_length,
            self.iota,
            self.iotaN,
            self.G0,
            self.helicity,
            self.X1c_untwisted,
            self.X1s_untwisted,
            self.Y1s_untwisted,
            self.Y1c_untwisted,
            self.normal_R,
            self.normal_phi,
            self.normal_z,
            self.binormal_R,
            self.binormal_phi,
            self.binormal_z,
            self.L_grad_B,
            self.inv_L_grad_B,
            self.torsion,
            self.curvature,
            self.varphi,
            self.R0p,
            self.Z0p,
        ) = self._legacy_tuple(solution)

    @property
    def dofs(self) -> jax.Array:
        """Mutable legacy degrees of freedom ordered ``rc, zs, etabar``."""

        return self._dofs

    @dofs.setter
    def dofs(self, new_dofs: ArrayLike) -> None:
        new_dofs = jnp.asarray(new_dofs)
        if new_dofs.ndim != 1 or new_dofs.size != 2 * self.nfourier + 1:
            raise ValueError(f"dofs must have shape ({2 * self.nfourier + 1},).")
        self._dofs = new_dofs
        self.rc = new_dofs[: self.nfourier]
        self.zs = new_dofs[self.nfourier : 2 * self.nfourier]
        self.etabar = new_dofs[-1]
        self._refresh()

    @property
    def x(self) -> jax.Array:
        """Alias for :attr:`dofs`, retained for ESSOS optimizers."""

        return self.dofs

    @x.setter
    def x(self, new_x: ArrayLike) -> None:
        self.dofs = new_x

    def _tree_flatten(self):
        children = (
            self.rc,
            self.zs,
            self.etabar,
            self.B0,
            self.sigma0,
            self.I2,
            self.B2c,
            self.p2,
        )
        auxiliary = {
            "nphi": self.nphi,
            "spsi": self.spsi,
            "sG": self.sG,
            "nfp": self.nfp,
            "order": self.order,
        }
        return children, auxiliary

    @classmethod
    def _tree_unflatten(cls, auxiliary, children):
        rc, zs, etabar, B0, sigma0, I2, B2c, p2 = children
        return cls(
            rc=rc,
            zs=zs,
            etabar=etabar,
            B0=B0,
            sigma0=sigma0,
            I2=I2,
            B2c=B2c,
            p2=p2,
            **auxiliary,
        )

    def calculate(self, rc: ArrayLike, zs: ArrayLike, etabar: ArrayLike):
        """Return the historical tuple, evaluated by the canonical core."""

        return self._legacy_tuple(self._canonical_solution(rc, zs, etabar))

    def B_covariant(self, points: ArrayLike) -> jax.Array:
        """First-order covariant Boozer components ``(B_r, B_theta, B_phi)``."""

        r, _, _ = jnp.asarray(points)
        return jnp.asarray((0.0, r * r * self.I2, self.G0))

    def B_contravariant(self, points: ArrayLike) -> jax.Array:
        """First-order contravariant Boozer components."""

        r, _, _ = jnp.asarray(points)
        Bphi = r * self.AbsB(points) / self.jacobian(points)
        return jnp.asarray((0.0, self.iotaN * Bphi, Bphi))

    def AbsB(self, points: ArrayLike) -> jax.Array:
        """First-order field strength in near-axis coordinates."""

        r, theta, _ = jnp.asarray(points)
        return self.B0 * (1 + r * self.etabar * jnp.cos(theta))

    def jacobian(self, points: ArrayLike) -> jax.Array:
        """First-order coordinate Jacobian."""

        r, _, _ = jnp.asarray(points)
        field_strength = self.AbsB(points)
        return r * self.B0 * (self.G0 + self.iota * self.I2) / field_strength**2

    def interpolated_array_at_point(self, array: ArrayLike, point: ArrayLike) -> jax.Array:
        """Periodically interpolate a sampled one-field-period array."""

        period = 2 * jnp.pi / self.nfp
        array = jnp.asarray(array)
        return jnp.interp(
            jnp.asarray(point),
            jnp.append(self.phi, period),
            jnp.append(array, array[0]),
            period=period,
        )

    def Frenet_to_cylindrical_1_point(
        self,
        phi0: ArrayLike,
        X_at_this_theta: ArrayLike,
        Y_at_this_theta: ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Map one displaced Frenet point to cylindrical coordinates."""

        sine = jnp.sin(phi0)
        cosine = jnp.cos(phi0)
        R0 = self.interpolated_array_at_point(self.R0, phi0)
        Z0 = self.interpolated_array_at_point(self.Z0, phi0)
        X = self.interpolated_array_at_point(X_at_this_theta, phi0)
        Y = self.interpolated_array_at_point(Y_at_this_theta, phi0)
        normal_R = self.interpolated_array_at_point(self.normal_R, phi0)
        normal_phi = self.interpolated_array_at_point(self.normal_phi, phi0)
        normal_z = self.interpolated_array_at_point(self.normal_z, phi0)
        binormal_R = self.interpolated_array_at_point(self.binormal_R, phi0)
        binormal_phi = self.interpolated_array_at_point(self.binormal_phi, phi0)
        binormal_z = self.interpolated_array_at_point(self.binormal_z, phi0)

        normal_x = normal_R * cosine - normal_phi * sine
        normal_y = normal_R * sine + normal_phi * cosine
        binormal_x = binormal_R * cosine - binormal_phi * sine
        binormal_y = binormal_R * sine + binormal_phi * cosine
        x = R0 * cosine + X * normal_x + Y * binormal_x
        y = R0 * sine + X * normal_y + Y * binormal_y
        z = Z0 + X * normal_z + Y * binormal_z
        return jnp.hypot(x, y), z, jnp.arctan2(y, x)

    def Frenet_to_cylindrical_residual_func(
        self,
        phi0: ArrayLike,
        phi_target: ArrayLike,
        X_at_this_theta: ArrayLike,
        Y_at_this_theta: ArrayLike,
    ) -> jax.Array:
        """Wrapped cylindrical-angle residual for a Frenet point."""

        _, _, phi = self.Frenet_to_cylindrical_1_point(
            phi0,
            X_at_this_theta,
            Y_at_this_theta,
        )
        difference = phi - phi_target
        return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))

    def residual_phi0_of_theta_varphi_func(
        self,
        phi0: ArrayLike,
        r: ArrayLike,
        theta: ArrayLike,
        varphi: ArrayLike,
    ) -> jax.Array:
        """Residual for inversion at fixed Boozer toroidal angle."""

        X = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi = self.Frenet_to_cylindrical_1_point(phi0, X, Y)
        nu0 = self.interpolated_array_at_point(self.varphi - self.phi, phi0)
        X1c = self.interpolated_array_at_point(self.X1c_untwisted, phi0)
        X1s = self.interpolated_array_at_point(self.X1s_untwisted, phi0)
        Y1c = self.interpolated_array_at_point(self.Y1c_untwisted, phi0)
        Y1s = self.interpolated_array_at_point(self.Y1s_untwisted, phi0)
        bR = self.interpolated_array_at_point(self.binormal_R, phi0)
        bZ = self.interpolated_array_at_point(self.binormal_z, phi0)
        nR = self.interpolated_array_at_point(self.normal_R, phi0)
        nZ = self.interpolated_array_at_point(self.normal_z, phi0)
        R0 = self.interpolated_array_at_point(self.R0, phi0)
        R0p = self.interpolated_array_at_point(self.R0p, phi0)
        Z0p = self.interpolated_array_at_point(self.Z0p, phi0)
        nu1c = X1c * (bR * Z0p - bZ * R0p) / R0 + Y1c * (nZ * R0p - nR * Z0p) / R0
        nu1s = X1s * (bR * Z0p - bZ * R0p) / R0 + Y1s * (nZ * R0p - nR * Z0p) / R0
        nu = nu0 + r * (nu1c * jnp.cos(theta) + nu1s * jnp.sin(theta))
        return phi + nu - varphi

    def phi_of_theta_varphi(
        self,
        r: ArrayLike,
        theta: ArrayLike,
        varphi: ArrayLike,
    ) -> jax.Array:
        """Invert the regular-coordinate map for cylindrical toroidal angle."""

        residual = lambda phi0: self.residual_phi0_of_theta_varphi_func(  # noqa: E731
            phi0,
            r,
            theta,
            varphi,
        )
        phi_on_axis, _ = implicit_dense_root(residual, jnp.asarray(varphi))
        X = r * (self.X1c_untwisted * jnp.cos(theta) + self.X1s_untwisted * jnp.sin(theta))
        Y = r * (self.Y1c_untwisted * jnp.cos(theta) + self.Y1s_untwisted * jnp.sin(theta))
        _, _, phi = self.Frenet_to_cylindrical_1_point(phi_on_axis, X, Y)
        return phi

    def Frenet_to_cylindrical(
        self,
        r: ArrayLike,
        ntheta: int = 20,
        phi_is_varphi: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Map a first-order surface over one field period."""

        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        toroidal_grid = self.phi

        def for_theta(theta_value):
            X = r * (
                self.X1c_untwisted * jnp.cos(theta_value)
                + self.X1s_untwisted * jnp.sin(theta_value)
            )
            Y = r * (
                self.Y1c_untwisted * jnp.cos(theta_value)
                + self.Y1s_untwisted * jnp.sin(theta_value)
            )

            def for_toroidal_angle(target):
                if phi_is_varphi:
                    residual = lambda phi0: self.residual_phi0_of_theta_varphi_func(  # noqa: E731
                        phi0,
                        r,
                        theta_value,
                        target,
                    )
                else:
                    residual = lambda phi0: self.Frenet_to_cylindrical_residual_func(  # noqa: E731
                        phi0,
                        target,
                        X,
                        Y,
                    )
                phi0, _ = implicit_dense_root(residual, target)
                R, Z, _ = self.Frenet_to_cylindrical_1_point(phi0, X, Y)
                return R, Z, phi0

            return jax.vmap(for_toroidal_angle)(toroidal_grid)

        return jax.vmap(for_theta)(theta)

    def to_Fourier(  # noqa: N802
        self,
        R_2D: ArrayLike,
        Z_2D: ArrayLike,
        nfp: int,
        mpol: int,
        ntor: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Convert a sampled stellarator-symmetric surface to Fourier data."""

        R_2D = jnp.asarray(R_2D)
        Z_2D = jnp.asarray(Z_2D)
        ntheta, nphi = R_2D.shape
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
        phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
        phi2d, theta2d = jnp.meshgrid(phi, theta, indexing="xy")
        m = jnp.arange(mpol + 1)
        n = jnp.arange(-ntor, ntor + 1)
        angle = (
            m[None, :, None, None] * theta2d[None, None, :, :]
            - n[:, None, None, None] * nfp * phi2d[None, None, :, :]
        )
        factor = 2 / (ntheta * nphi)
        RBC = factor * jnp.sum(R_2D[None, None, :, :] * jnp.cos(angle), axis=(-2, -1))
        ZBS = factor * jnp.sum(Z_2D[None, None, :, :] * jnp.sin(angle), axis=(-2, -1))
        RBC = RBC.at[ntor, 0].set(jnp.mean(R_2D))
        RBC = RBC.at[:ntor, 0].set(0)
        ZBS = ZBS.at[:ntor, 0].set(0)
        return RBC, ZBS

    def get_boundary(
        self,
        r: ArrayLike = 0.1,
        ntheta: int = 30,
        nphi: int = 120,
        ntheta_fourier: int = 20,
        mpol: int = 5,
        ntor: int = 5,
        phi_is_varphi: bool = False,
        phi_offset: ArrayLike = 0.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return a full-torus first-order surface in Cartesian coordinates."""

        R_period, Z_period, _ = self.Frenet_to_cylindrical(
            r,
            ntheta=ntheta_fourier,
            phi_is_varphi=phi_is_varphi,
        )
        RBC, ZBS = self.to_Fourier(R_period, Z_period, self.nfp, mpol, ntor)
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta)
        original_phi = jnp.linspace(0, 2 * jnp.pi, nphi) + phi_offset
        phi2d, theta2d = jnp.meshgrid(original_phi, theta, indexing="xy")

        if phi_is_varphi:
            phi2d = jax.vmap(
                lambda theta_row, varphi_row: jax.vmap(
                    lambda theta_value, varphi_value: self.phi_of_theta_varphi(
                        r,
                        theta_value,
                        varphi_value,
                    )
                )(theta_row, varphi_row)
            )(theta2d, phi2d)

        m = jnp.arange(mpol + 1)
        n = jnp.arange(-ntor, ntor + 1)
        angle = (
            m[None, :, None, None] * theta2d[None, None, :, :]
            - n[:, None, None, None] * self.nfp * original_phi[None, None, None, :]
        )
        R = jnp.sum(RBC[:, :, None, None] * jnp.cos(angle), axis=(0, 1))
        Z = jnp.sum(ZBS[:, :, None, None] * jnp.sin(angle), axis=(0, 1))
        return R * jnp.cos(phi2d), R * jnp.sin(phi2d), Z, R

    def B_mag(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> jax.Array:
        """First-order field strength using the legacy angle convention."""

        return self.B0 * (1 + r * self.etabar * jnp.cos(theta - (self.iota - self.iotaN) * phi))

    def plot(
        self,
        r: float = 0.1,
        ntheta: int = 40,
        nphi: int = 120,
        ntheta_fourier: int = 20,
        ax=None,
        show: bool = True,
        close: bool = False,
        axis_equal: bool = True,
        **kwargs,
    ):
        """Plot the first-order boundary without importing ESSOS."""

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm
        from matplotlib.colors import LightSource, Normalize

        created_axes = ax is None or getattr(ax, "name", None) != "3d"
        if created_axes:
            figure = plt.figure()
            ax = figure.add_subplot(projection="3d")
        else:
            figure = ax.figure

        x, y, z, _ = self.get_boundary(
            r=r,
            ntheta=ntheta,
            nphi=nphi,
            ntheta_fourier=ntheta_fourier,
        )
        theta = jnp.linspace(0, 2 * jnp.pi, ntheta)
        phi = jnp.linspace(0, 2 * jnp.pi, nphi)
        phi2d, theta2d = jnp.meshgrid(phi, theta)
        field_strength = np.asarray(self.B_mag(r, theta2d, phi2d))
        normalization = Normalize(vmin=field_strength.min(), vmax=field_strength.max())
        colormap = cm.viridis
        facecolors = LightSource(azdeg=0, altdeg=10).shade(
            field_strength,
            colormap,
            norm=normalization,
        )
        kwargs.setdefault("alpha", 1)
        ax.plot_surface(
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            antialiased=False,
            linewidth=0,
            shade=False,
            **kwargs,
        )
        if created_axes:
            colorbar = figure.colorbar(
                cm.ScalarMappable(cmap=colormap, norm=normalization),
                ax=ax,
                shrink=0.7,
            )
            colorbar.ax.set_title(r"$|B|$ [T]")
            ax.grid(False)
        if axis_equal:
            ranges = (
                np.ptp(np.asarray(x)),
                np.ptp(np.asarray(y)),
                np.ptp(np.asarray(z)),
            )
            radius = max(ranges) / 2
            centers = (
                np.mean(np.asarray(x)),
                np.mean(np.asarray(y)),
                np.mean(np.asarray(z)),
            )
            ax.set_xlim(centers[0] - radius, centers[0] + radius)
            ax.set_ylim(centers[1] - radius, centers[1] + radius)
            ax.set_zlim(centers[2] - radius, centers[2] + radius)
        if show:
            plt.show()
        if close:
            plt.close(figure)
        return figure, ax


jax.tree_util.register_pytree_node(
    near_axis,
    near_axis._tree_flatten,
    near_axis._tree_unflatten,
)
