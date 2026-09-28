"""Reference near-axis configurations used by the tests.

Sources:
- ``r1_qa``: Landreman, Sengupta & Plunk, J. Plasma Phys. 85, 905850103 (2019), section 5.1
  (pyQSC ``"r1 section 5.1"``).
- ``qa``, ``qh`` and ``finite_pressure_current``: Landreman & Sengupta, J. Plasma Phys. 85,
  815850601 (2019), sections 5.1, 5.2 and 5.5 (pyQSC ``"r2 section 5.1/5.2/5.5"``).
- ``database_*`` and ``plasma_stellarator``: stellarator database of Curvo et al.
  (2025), https://stellarator.physics.wisc.edu/app/plot/<id>.
- ``b20_optimized_good``: eight-mode QH refinement of database configuration 57409 by
  exact B2c elimination and bounded least squares (preserved optimizer on
  ``origin/preserve/pr2-before-refactor-2026-09-27``).
- ``plasma_dominant_channel``: synthetic finite-beta circular channel used for
  plasma-field validation.
"""

from __future__ import annotations

from typing import Any

import pyqsc_jax as qsc

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "r1_qa": dict(rc=(1.0, 0.045), zs=(0.0, -0.045), nfp=3, etabar=-0.9),
    "qh": dict(
        rc=(1.0, 0.17, 0.01804, 0.001409, 5.877e-5),
        zs=(0.0, 0.1581, 0.01820, 0.001548, 7.772e-5),
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
    ),
    "qa": dict(rc=(1.0, 0.155, 0.0102), zs=(0.0, 0.154, 0.0111), nfp=2, etabar=0.64, B2c=-0.00322),
    "finite_pressure_current": dict(
        rc=(1.0, 0.09), zs=(0.0, -0.09), nfp=2, etabar=0.95, B2c=-0.7, I2=0.9, p2=-600000.0
    ),
    "plasma_dominant_channel": dict(rc=(1.0,), zs=(0.0,), nfp=1, etabar=0.5, I2=4.2, p2=-100000.0),
    "plasma_stellarator": dict(  # database id 52521
        rc=(1.0, -0.5415884, 0.029195854, 0.0048646266),
        zs=(0.0, -0.57113713, 0.029922731, 0.0041398546),
        nfp=4,
        etabar=1.1396117,
        B2c=-0.050057083,
        p2=-28248.188,
        order="r3",
    ),
    "database_qa_139524": dict(
        rc=(1.0, -0.06883207, 0.0017516185, 0.023231717),
        zs=(0.0, -0.28447896, 0.074662544, 0.07483574),
        nfp=1,
        etabar=-0.7771866,
        B2c=-1.8120022,
        p2=-232743.75,
        order="r3",
    ),
    "database_example_3": dict(
        rc=(1.0, -0.53677857, -0.046455786, -0.0070183445),
        zs=(0.0, -0.5888703, -0.04447083, -0.009581006),
        nfp=4,
        etabar=1.4014399,
        B2c=-0.7512066,
        p2=-74635.375,
        order="r3",
    ),
    "database_large_singularity_107579": dict(
        rc=(1.0, -0.54465365, 0.005908036, 0.0054288576),
        zs=(0.0, 0.540987, -0.009328544, -0.003327578),
        nfp=3,
        etabar=-0.95917624,
        B2c=0.10977107,
        p2=-19211.062,
        order="r3",
    ),
    "database_low_b20_57409": dict(
        rc=(1.0, -0.51677144, -0.009499784, -0.005914526),
        zs=(0.0, -0.5420635, -0.012225689, -0.0059485724),
        nfp=4,
        etabar=-1.3295174,
        B2c=-0.7577404,
        p2=-23501.281,
        order="r3",
    ),
    "b20_optimized_good": dict(
        rc=(
            1.0,
            -0.5039436500066075,
            -0.043965867334349464,
            -0.00654919263283669,
            -3.047898633100702e-06,
            2.7519909478191202e-05,
            6.071324626161564e-06,
            8.46692978641554e-07,
            5.99743474381913e-08,
        ),
        zs=(
            0.0,
            -0.5050020451312105,
            -0.045010140721391825,
            -0.006585874304053245,
            -1.5550078876295003e-05,
            2.6653739413147386e-05,
            5.978859527401134e-06,
            8.327402553082346e-07,
            5.9109366390399247e-08,
        ),
        nfp=4,
        etabar=-1.3295174,
        B2c=-1.132420959333329,
        p2=-23501.281,
        order="r3",
    ),
}


def solve_configuration(name: str, **overrides: Any) -> qsc.NearAxisSolution:
    """Solve a named reference configuration (default order ``r2``)."""

    parameters = {"order": "r2", **CONFIGURATIONS[name], **overrides}
    return qsc.Qsc(**parameters)
