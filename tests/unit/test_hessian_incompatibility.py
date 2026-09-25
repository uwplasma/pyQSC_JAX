"""Local Hessian incompatibility identity for the rank-three STF projector.

With ``H[i, j, k] = d_j d_k B_i`` and ``C[k, l] = eps[k, j, i] H[i, j, l]``
(``= mu0 d_l J_k``), a divergence-free field satisfies

    ||H - STF(H)||**2 = (2/3)||Sym C||**2 + (4/5)||Skew C||**2.

This is a local lower bound on how far a current-carrying Hessian is from any
vacuum target; projecting is not the same as subtracting the plasma field.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space

from pyqsc_jax.plasma import project_symmetric_trace_free_rank3

MU0 = 4e-7 * np.pi
EPS = np.zeros((3, 3, 3))
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[i, j, k], EPS[j, i, k] = 1.0, -1.0


def _curl_tensor(H):
    return np.einsum("kji,ijl->kl", EPS, H)


def _identity_sides(H):
    C = _curl_tensor(H)
    symmetric, skew = (C + C.T) / 2, (C - C.T) / 2
    left = np.sum((H - np.asarray(project_symmetric_trace_free_rank3(H))) ** 2)
    return left, 2 / 3 * np.sum(symmetric**2) + 4 / 5 * np.sum(skew**2)


def test_identity_on_random_divergence_free_hessians():
    basis = []
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                E = np.zeros((3, 3, 3))
                E[i, j, k] = E[i, k, j] = 1.0
                basis.append(E.ravel())
    basis = np.array(basis).T
    divergence = np.array(
        [[np.einsum("iik->k", b.reshape(3, 3, 3))[k] for b in basis.T] for k in range(3)]
    )
    free = basis @ null_space(divergence)
    assert free.shape[1] == 15
    rng = np.random.default_rng(7)
    for _ in range(200):
        H = (free @ rng.normal(size=free.shape[1])).reshape(3, 3, 3)
        left, right = _identity_sides(H)
        np.testing.assert_allclose(left, right, rtol=1e-12)
    # A vacuum (curl- and divergence-free) Hessian is already STF.
    vacuum = np.asarray(project_symmetric_trace_free_rank3(H))
    left, right = _identity_sides(vacuum)
    assert left < 1e-24 and right < 1e-24


def test_cylinder_with_linear_current():
    j1 = 3.0e5  # A/m^3: J_z = j1 * x, divergence free

    def potential(p):
        # A_z = -mu0 j1 x (x^2 + y^2) / 8 solves lap A_z = -mu0 J_z.
        return -MU0 * j1 * p[0] * (p[0] ** 2 + p[1] ** 2) / 8

    def field(x):
        g = jax.grad(potential)(x)
        return jnp.array([g[1], -g[0], 0.0])

    point = jnp.array([0.013, -0.004, 0.2])
    H = np.asarray(jax.jacfwd(jax.jacfwd(field))(point))
    np.testing.assert_allclose(np.einsum("iik->k", H), 0, atol=1e-18)
    C = _curl_tensor(H)
    expected = np.zeros((3, 3))
    expected[2, 0] = MU0 * j1  # mu0 d_x J_z
    np.testing.assert_allclose(C, expected, atol=1e-15)
    left, right = _identity_sides(H)
    np.testing.assert_allclose(left, right, rtol=1e-12)
    # C has one entry, so (2/3 + 4/5)/2 of its square: the distance to the
    # nearest vacuum Hessian is fixed by the current gradient alone.
    np.testing.assert_allclose(left, 11 / 15 * (MU0 * j1) ** 2, rtol=1e-12)
