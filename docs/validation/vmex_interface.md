# Differentiable VMEX integration

The bridge was audited and exercised against
[uwplasma/VMEX](https://github.com/uwplasma/vmex) version 0.3.0 at commit
`2a40d7566be083070ea3ea534fa5d1fc44ad733a`. A compatibility CI job checks
the interface against current VMEX `main` on every pyQSC_JAX pull request.

## Tested contract

The integration uses VMEX's public APIs:

- `VmecInput`;
- `implicit.params_from_input` and `implicit.run`;
- `implicit.iota_profile`;
- `optimize.QuasisymmetryRatioResidual.profile_state`;
- `optimize.magnetic_well`, `aspect_ratio`, and `volume`.

The fixed-boundary solve and derived quantities remain inside the JAX
transformation. Unit tests also differentiate through
`problem.parameters_for(candidate_solution)`, which includes pyQSC_JAX's
surface conversion and pressure/current normalization.
The same diagnosed cylindrical-angle conversion used by `to_vmec` is used
in-memory. Concrete nonconvergence raises, while a traced optimization that
leaves the valid conversion domain receives nonfinite boundary leaves instead
of a silently incorrect equilibrium.

The live compatibility test covers:

| Case | Pressure | Current | Axis | Checked quantities |
| --- | ---: | ---: | --- | --- |
| vacuum QA | zero | zero | nonplanar QA | \(\iota(s)\), QS profile, well, implicit boundary gradient |
| `plasma_stellarator` (database ID 52521) | finite | exactly zero | nonplanar, RMS torsion \(0.979\ \mathrm{m}^{-1}\) | \(\iota(s)\), QS profile, well, nonzero thermal energy, implicit pressure/boundary gradients |

Both live cases, including their implicit gradients, passed locally against
the exact current VMEX `main` commit above on 2026-07-30. The same source
checkout is installed by the compatibility CI job rather than relying on a
stale wheel.

At the deliberately small `ns=7`, `mpol=4`, `ntor=2` smoke resolution, the
vacuum VMEX axis value in the pyQSC_JAX sign convention is `-0.421520`,
compared with the near-axis value `-0.420473`. This is a 0.249% difference;
the test gate is 0.8%. The purpose of this low-resolution job is cross-package
AD compatibility, not a production equilibrium accuracy claim.

The strongly shaped finite-beta adjoint uses an explicit
`adjoint_tol=1e-8`; the public API retains VMEX's stricter \(10^{-11}\)
default unless the caller makes that tradeoff explicitly. VMEX enforces the
selected residual tolerance and raises rather than returning an unconverged
adjoint.

The publication example records finite-beta values of:

| Quantity | Value |
| --- | ---: |
| magnetic well | \(1.84561\times10^{-2}\) |
| \(\partial W/\partial p_\mathrm{scale}\) | \(+6.22590\times10^{-5}\) |
| \(\|\partial W/\partial RBC\|_2\) | \(9.78080\times10^1\) |

These values are finite, nonzero regression evidence that the pressure and
boundary adjoint paths are active.

## Coordinate convention

VMEC's native transform for boundaries written by pyQSC_JAX has the opposite
sign from pyQSC_JAX's toroidal-angle convention. `VmexRadialQuantities`
therefore exposes both:

- `iota_vmec`: VMEX native sign;
- `iota`: the sign aligned with `NearAxisSolution.iota`.

The conversion is explicit and covered by tests.

## Scope limits

- The bridge differentiates VMEX's fixed-boundary equilibrium. It does not
  claim an adjoint of the reconverged free-boundary NESTOR root.
- VMEX's traceable quasisymmetry profile currently requires stellarator
  symmetry. Asymmetric equilibria can be solved with `qs_surfaces=()`.
- The magnetic well is VMEX's canonical endpoint scalar. No undocumented
  radial “well profile” is invented.
- Pressure/current profiles are the simple near-axis-consistent profiles used
  by `to_vmec`; users can replace the returned VMEX parameter leaves when a
  different finite-radius profile model is intended.
- Production results require radial, angular, and Fourier-resolution studies
  beyond the fast compatibility settings above.
