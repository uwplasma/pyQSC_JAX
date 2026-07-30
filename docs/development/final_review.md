# Final hostile review

This review treats every sign, convention, compatibility promise, solver
status, dependency, and global-search claim as suspect until supported by an
equation, an independent comparison, or a failure-sensitive test.

## Disposition

No known pyQSC_JAX correctness defect remains after the local final gate.
The branch is suitable for draft-PR review, not for an immediate production
release. Production publication still requires remote CI, PR approval,
TestPyPI trusted-publishing verification, release-version metadata, and a
Zenodo archive/DOI.

## Evidence

| Gate | Result |
| --- | --- |
| Ruff lint and format | clean over `src`, `tests`, `examples`, `benchmarks` |
| Full test/coverage run | 287 passed, 3 integration skips; 98.79% combined line/branch coverage |
| Documentation | HTML and doctest builders pass with warnings as errors |
| Examples | 14 tutorials + 10 publication scripts pass from clean cwd |
| Package metadata | wheel and sdist pass `twine check` |
| Clean artifacts | wheel and independently rebuilt sdist pass physics smokes |
| Dependencies | both environments pass `pip check`; no known advisories |
| High-resolution parity | QA/finite/QH r2 max array delta \(1.11\times10^{-11}\) at `nphi=121` |
| ESSOS clean integration | 12 field-jet/objective/example tests pass |
| Mutation: sigma sign | upstream-reference test fails with 200% relative transform error |
| Mutation: external Hessian `-`→`+` | STF symmetry test fails with order-unity error |

The dependency auditor necessarily skips the unreleased `pyqsc-jax`
distribution itself because it has no PyPI record. It does audit the installed
JAX/SOLVAX dependency tree.

## Convention audit

- Axis Fourier packing is exactly `rc, rs, zc, zs`; constructor and legacy
  `dofs` setter round-trip in the same order.
- Canonical sample axes lead. Gradient and Hessian are field-component-first:
  `D[n,i,j] = dB_i/dx_j` and
  `H[n,i,j,k] = d²B_i/(dx_j dx_k)`.
- Cylindrical vector differentiation includes basis-rotation connection
  terms; fixed Cartesian arrays are not incorrectly treated as
  field-periodic.
- Torsion follows Landreman--Sengupta and is documented as opposite the
  original Garren--Boozer convention.
- Frame helicity, `sG`, and `spsi` enter
  `iotaN = iota + N*nfp` consistently in forward, inverse, and continuation
  systems.
- Current conversion uses `chi = sG*spsi`; a positive formal radius is
  enforced before enclosed-current conversion or plasma-field matching.
- External jets are `total - plasma`. The 5- and 7-component packers project
  and reconstruct the documented Cartesian STF bases.
- `verified_zero` is restricted to the known zero lower bound of the
  nonnegative primary \(B_{20}\) residual. Nonzero searches remain
  `best_found`.

## Solver and AD audit

- The sigma root is convergence-tested; the old fixed-five-step path is gone.
- The r2 coupled system reports residual and condition number.
- SOLVAX differentiates converged root/linear equations, not Newton or dense
  solver iteration histories.
- Nonconverged candidates retain structured reports and are never silently
  certified.
- JIT, VMAP, JVP, VJP, and finite-difference tests cover first order, r2,
  inverse branches, optimization, and plasma jets.
- Discrete basin selection is not advertised as differentiable; smooth
  candidate residuals are.

## Architecture and dependency audit

- Physics lives in small immutable pytree/dataclass records and pure
  functions; the mutable class is only a compatibility adapter.
- The package has no mutable global physics state and performs no import-time
  JAX configuration.
- Runtime dependencies are only unpinned `jax` and `solvax`. Equinox remains
  an indirect SOLVAX implementation dependency rather than a public modeling
  requirement.
- The unused `pyevtk` plotting extra found during review was removed.
- No `setup.py`, separate `jaxlib` declaration, generic `utils.py`, or ESSOS
  dependency remains.
- The wheel contains only package code and required license/NOTICE metadata;
  the sdist contains sources, tests, docs, examples, benchmark report, and
  citation files.

## Documentation/API findings fixed during review

- Replaced an unsafe tag-only production publish path with reviewed-release,
  full-verification, main-ancestry, OIDC TestPyPI/PyPI jobs.
- Added missing Codecov, docs-example, clean ESSOS, and benchmark artifact
  workflow coverage.
- Removed stale documentation names `V1r` and `B2cQI`, which are not public r2
  fields.
- Corrected the publication surface plot so the magnetic-axis overlay spans
  the full torus.
- Kept the DOI badge explicitly pending instead of fabricating a Zenodo
  identifier.

## Exact remaining limitations

- Near-axis and plasma-channel results are asymptotic; field/Hessian remainder
  values are scales, not rigorous bounds.
- The plasma Hessian is validated through its independently derived interior
  contact terms and Maxwell identities; differentiating a singular filament
  quadrature is not a valid independent reference.
- Frenet geometry fails at zero curvature.
- `etabar` continuation has a fixed step and does not yet provide a generic
  multiparameter continuation engine.
- Magnetic shear supports only the documented standard-MHS sign
  specialization.
- A finite multistart budget cannot certify a nonzero global minimum.
- Fast-particle confinement, finite-radius equilibrium, and coil engineering
  are external validations.
- The ESSOS base branch retains unrelated collection and documentation
  failures described in ESSOS PR #46; the new field-jet slice is independently
  green.
- TestPyPI upload, PyPI trusted-publisher configuration, PR approval, and
  Zenodo DOI creation require maintainer/external service actions and are not
  represented as complete.
