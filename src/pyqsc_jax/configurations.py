"""Named, immutable near-axis reference configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyqsc_jax.first_order import Qsc
from pyqsc_jax.models import NearAxisSolution


@dataclass(frozen=True)
class ReferenceConfiguration:
    """A documented set of inputs used in regression and example workflows."""

    name: str
    description: str
    rc: tuple[float, ...]
    zs: tuple[float, ...]
    nfp: int
    etabar: float
    B0: float = 1.0
    B2c: float = 0.0
    I2: float = 0.0
    p2: float = 0.0
    order: str = "r2"

    def parameters(self, **overrides: Any) -> dict[str, Any]:
        """Return fresh keyword arguments, with explicit caller overrides."""

        parameters: dict[str, Any] = {
            "rc": self.rc,
            "zs": self.zs,
            "nfp": self.nfp,
            "etabar": self.etabar,
            "B0": self.B0,
            "B2c": self.B2c,
            "I2": self.I2,
            "p2": self.p2,
            "order": self.order,
        }
        parameters.update(overrides)
        return parameters

    def solve(self, **overrides: Any) -> NearAxisSolution:
        """Solve this configuration with optional input overrides."""

        return Qsc(**self.parameters(**overrides))


REFERENCE_CONFIGURATIONS = (
    ReferenceConfiguration(
        name="qa",
        description="Vacuum quasi-axisymmetric reference used for pyQSC parity.",
        rc=(1.0, 0.155, 0.0102),
        zs=(0.0, 0.154, 0.0111),
        nfp=2,
        etabar=0.64,
        B2c=-0.00322,
    ),
    ReferenceConfiguration(
        name="qh",
        description="Vacuum quasi-helically symmetric reference with nonzero frame helicity.",
        rc=(1.0, 0.17, 0.01804, 0.001409, 5.877e-5),
        zs=(0.0, 0.1581, 0.01820, 0.001548, 7.772e-5),
        nfp=4,
        etabar=1.569,
        B2c=0.1348,
    ),
    ReferenceConfiguration(
        name="finite_pressure_current",
        description="Finite-pressure/current r2 reference used throughout plasma-jet validation.",
        rc=(1.0, 0.09),
        zs=(0.0, -0.09),
        nfp=2,
        etabar=0.95,
        B2c=-0.7,
        I2=0.9,
        p2=-600000.0,
    ),
    ReferenceConfiguration(
        name="b20_optimized_qa",
        description=(
            "High-order QA axis refined by exact B2c elimination and bounded "
            "least-squares minimization of the dense B20 anomaly."
        ),
        rc=(
            1.0038581971135636,
            0.15364113635027635,
            0.019066219667598507,
            0.0029880230763716796,
            0.00040790711310020265,
            0.00004610617355685316,
            4.102693907398271e-06,
            5.154300428457222e-07,
            4.8802742243232844e-08,
            7.301132037525988e-09,
        ),
        zs=(
            0.0,
            -0.14606422619055753,
            -0.020369711261517005,
            -0.003031761163968464,
            -0.00039984211636995954,
            -0.00005034787759957322,
            -4.174376962124085e-06,
            -4.557462755956434e-07,
            -8.173481495049928e-08,
            -3.732477282851326e-09,
        ),
        nfp=2,
        etabar=-0.6783912804454629,
        B0=1.006541121335688,
        B2c=0.9427285320639192,
    ),
    ReferenceConfiguration(
        name="plasma_dominant_channel",
        description=(
            "Circular finite-pressure/current channel for which the matched plasma "
            "field exceeds 30 percent of the total field at formal radius 0.2."
        ),
        rc=(1.0,),
        zs=(0.0,),
        nfp=1,
        etabar=0.5,
        I2=4.2,
        p2=-100000.0,
    ),
)


def available_configurations() -> tuple[str, ...]:
    """Return the stable names of bundled reference configurations."""

    return tuple(configuration.name for configuration in REFERENCE_CONFIGURATIONS)


def get_configuration(name: str) -> ReferenceConfiguration:
    """Return one named reference configuration."""

    for configuration in REFERENCE_CONFIGURATIONS:
        if configuration.name == name:
            return configuration
    available = ", ".join(available_configurations())
    raise ValueError(f"Unknown configuration {name!r}. Available configurations: {available}.")


def solve_configuration(name: str, **overrides: Any) -> NearAxisSolution:
    """Solve a named reference configuration."""

    return get_configuration(name).solve(**overrides)
