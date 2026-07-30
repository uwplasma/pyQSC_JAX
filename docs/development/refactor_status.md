# Refactor status

This log records green phase boundaries on
`refactor/pyqsc-jax-complete`. “Not benchmarked” means no performance claim
was made for that phase.

## Phase 1 — audit and compatibility baseline

- Commit: `3a87ba6`.
- Files: baseline report, architecture/solver ADRs, local reference policy,
  compatibility regression tests.
- Verification: legacy imports, shapes, mutability, boundary behavior, and
  upstream reference extraction.
- Coverage/residuals: baseline only; inherited five-step sigma behavior
  recorded rather than accepted.
- Performance/API: not benchmarked; historical API frozen.
- Risk/next: mutable monolith and inconsistent DOF unpacking; build immutable
  axis/geometry core.

## Phase 2 — packaging and quality scaffold

- Commits: `5895008`, `969d616`.
- Files: `pyproject.toml`, `src/` layout, CI, docs configuration, license and
  citation metadata.
- Verification: clean editable install, wheel/sdist smoke, Ruff, x64 CI.
- Coverage/residuals: 95% line/branch gates established.
- Performance/API: no import-time JAX policy; `jaxlib` removed from metadata.
- Risk/next: physics still incomplete; establish spectral geometry and a
  converged first-order solve.

## Phase 3 — axis, geometry, and first order

- Commits: `28955f8`, `16249fe`.
- Files: immutable axis/geometry/spectral modules, damped sigma solver,
  first-order models, thin compatibility adapter.
- Verification: analytic circle, asymmetric reference geometry, QA/QH and
  finite-current parity, JIT/VMAP/JVP/VJP/finite differences.
- Coverage/residuals: QA sigma residual \(1.2\times10^{-14}\); coverage above
  the 95% gate.
- Performance/API: explicit pytrees replace static mutable `self`; no
  benchmark claim.
- Risk/next: r2 outputs unavailable; implement the complete coupled system.

## Phase 4 — complete r2 and diagnostics

- Commits: `3978a6b`, `5e67237`, `0086d77`.
- Files: second-order solve, total regular-coordinate Hessian, Mercier and
  singular-radius diagnostics.
- Verification: four independent r2 residual equations; vacuum QA,
  finite-pressure/current, and QH upstream arrays; Maxwell identities;
  singular-map roots; JIT and AD checks.
- Coverage/residuals: finite-current linear residual
  \(1.6\times10^{-13}\); field reconstruction at roundoff.
- Performance/API: one implicit dense linear solve; not benchmarked.
- Risk/next: third-order boundary/shear parity; implement as a separate green
  milestone.

## Phase 5 — third order and shear

- Commits: `99e81d5`, `935e1f8`.
- Files: r3 flux constraint, untwisted boundary harmonics, standard-MHS
  magnetic shear.
- Verification: two flux-constraint forms; QA/QH/finite-current upstream
  coefficients and boundaries; three shear cases; resolution and AD.
- Coverage/residuals: consistency residuals meet documented test tolerances;
  coverage above 95%.
- Performance/API: shear remains explicit because `B31c` is independent.
- Risk/next: shear sign specialization is documented; add inverse branches
  and folds.

## Phase 6 — inverse solves and optimization

- Commits: `7e87d04`, `abcbd0a`, `5169173`, `cf3f6da`, `6fab74b`.
- Files: target-transform inverse, full-state continuation, \(B_{20}\)
  diagnostics, analytic `B2c`, criteria, bounded multistart axis search.
- Verification: branch round trips, two inverse basins, fold traversal,
  affine reconstruction \(3.3\times10^{-15}\), exact scalar stationarity,
  criteria scaling, basin/status/resolution tests.
- Coverage/residuals: complete suite remained above 95% branch coverage.
- Performance/API: deterministic batched coarse search and JAX Jacobians;
  absolute timing not benchmarked.
- Risk/next: nonzero results explicitly lack global certificates; implement
  the surface-free current and field.

## Phase 7 — plasma source and external 3+5+7 jet

- Commits: `d46b2bc`, `f173de7`, `f7d15b3`, `c7277fc`.
- Files: positive-volume current, full-torus matched field, local gradient and
  Hessian, STF pack/unpack, asymptotic metadata.
- Verification: current conversions, straight/circular/elliptical channels,
  resolved-volume Biot--Savart field scaling, Ampère/divergence, QA/QH
  external STF identities, vacuum reduction, convergence, JIT/JVP/FD.
- Coverage/residuals: 203 tests passed; 98.90% branch-aware coverage;
  `plasma.py` 100%.
- Performance/API: surface-free calls accept explicit formal radius; not
  benchmarked.
- Risk/next: Hessian contact terms rely on the independently derived interior
  potential; integrate only the external target into ESSOS.

## Phase 8 — ESSOS integration

- ESSOS commits: `8a1ce7f`, `d20c23d`; draft PR
  [#46](https://github.com/uwplasma/ESSOS/pull/46).
- Files: external target/residual models, coil Hessian interface, stage-two
  and single-stage examples, cross-repository tests.
- Verification: 24 focused tests; actual coil and near-axis gradients versus
  finite differences; vacuum reduction; example objectives 10.74→4.09 and
  10.75→4.52.
- Coverage/performance: focused integration only; example budgets are smoke
  demonstrations, not device-quality searches.
- API: dependency remains `ESSOS -> pyQSC_JAX`.
- Risk/next: inherited ESSOS base-suite collection/docs failures are disclosed
  in the PR; complete pyQSC_JAX examples and documentation.

## Phase 9 — examples and publication figures

- Commit: `a8eb51a`.
- Files: named configurations, lazy plotting API, 12 tutorials, 5
  deterministic publication scripts, clean-cwd execution tests.
- Verification: 23 focused tests; every script produced its figure; all five
  publication plots visually inspected; Ruff clean.
- Coverage/residuals: tutorial AD relative error \(9.4\times10^{-10}\);
  tutorial continuation crosses a detected fold.
- Performance/API: full clean-directory example suite took 152 s on the
  development Mac; this includes separate-process JAX compilation and is not a
  package benchmark.
- Risk/next: finish docs, release hardening, clean artifact validation, and
  hostile review.

## Phase 10 — documentation and release hardening

- Commit: `69b05da`.
- Files: complete docs hierarchy, tutorials via `literalinclude`, migration,
  validation reports, release checklist, README, and compact reproducible
  figures.
- Verification: warnings-as-errors HTML and doctest builds; 23 focused
  configuration/plot/example tests; figure visual inspection.
- Performance/API: named configurations and lazy plotting helpers added; no
  new physics formula.
- Risk/next: harden coverage, packaging, publishing, compatibility, and
  benchmark workflows before final review.

## Phase 11 — release hardening

- Commit: `686be1e`.
- Files: synchronized benchmark and raw report, CI coverage/Codecov, docs
  example smoke, ESSOS compatibility job, TestPyPI/production OIDC workflow,
  clean wheel/sdist verification.
- Verification: 226 tests pass with 98.64% combined line/branch coverage;
  HTML/doctest warning-free; wheel and sdist install independently; `pip
  check` and dependency audit clean; 12 ESSOS field-jet tests pass in a fresh
  environment.
- Numerical checks: direct `nphi=121` QA/finite/QH comparison with audited
  pyQSC has maximum r2 array difference \(1.11\times10^{-11}\).
- Performance: 61-point first-order solve 0.198 s cold and 144 µs warm on the
  documented Apple M4 run; raw samples and caveats are committed.
- API: removed unused `pyevtk`; core dependencies remain only `jax` and
  `solvax`.
- Risk/next: remote CI, TestPyPI OIDC, PR approval, and Zenodo are external
  release gates; perform hostile final review before requesting merge.

## Phase 12 — final review

- Commit: `fb7ddac`.
- Evidence: convention/API/dependency audits complete; two representative
  physics mutations are killed by targeted tests; exact limitations and
  external release gates are recorded in `final_review.md`.
- Verification: final local lint, docs, clean-state, artifact, upstream, and
  ESSOS checks pass. The latest remote CI result is reported in the PR rather
  than hard-coded into this versioned status page.

## Phase 13 — high-signal examples and VMEC equilibrium validation

- Files: vectorized VMEC exporter, deterministic INDATA writer, frozen VMEC
  9.0 equilibrium, B20/plasma-dominant named cases, optimizer/export
  benchmarks, publication figures, and expanded validation docs.
- Verification: 255 tests pass with 98.74% combined line/branch coverage;
  `vmec.py` has 100% line/branch coverage; warnings-as-errors documentation
  builds; both frozen and live local-VMEC equilibrium checks pass.
- Numerical checks: optimized QA weighted B20 residual
  \(1.5901\times10^{-6}\) at `nphi=121` and stable through `nphi=481`;
  plasma-field fraction 33.26%; VMEC on-axis-iota error 0.0565% at radius
  0.0025 with force residuals below \(7.6\times10^{-11}\).
- Performance: synchronized VMEC conversion is 0.476 s including compilation
  and 5.20 ms warm at the documented high-resolution boundary setting.
- Risk/next: on-axis-iota agreement is asymptotic and degrades at larger
  export radius as documented; remote CI and review remain external gates.

## Phase 14 — VMEX radial equilibria and visual design audit

- Files: optional VMEX bridge, live current-main compatibility job, radial
  equilibrium tutorial and publication figure, full-torus 3D plotting
  helpers, four-case gallery, and angle-dependent plasma configuration.
- Verification: vacuum and finite-beta VMEX solves; radial \(\iota\), QS,
  magnetic-well, energy, and boundary-gradient checks; traceable
  `parameters_for` differentiation; focused plotting and plasma regressions.
- Numerical checks: low-resolution vacuum VMEX/near-axis on-axis-\(\iota\)
  difference 0.249%; finite-beta magnetic well
  \(-4.2297\times10^{-4}\); angle-dependent minimum plasma-field fraction
  32.70%; optimized \(B_{20}\) surface shown 3.3 times inside its singular
  radius.
- Performance: the VMEX smoke benchmark separates the converged forward solve
  from the magnetic-well value-and-implicit-gradient cost; it is documented
  as an integration timing, not a production throughput claim.
- Risk/next: VMEX quasisymmetry diagnostics currently require stellarator
  symmetry and the bridge covers fixed-boundary implicit differentiation, not
  a reconverged free-boundary NESTOR adjoint. Remote CI and review remain
  external gates.

## Phase 15 — screened database showcase and constrained \(B_{20}\)

- Files: traceable configurations from Wisconsin database IDs 3, 57409, and
  107579; an eight-mode constrained refinement of ID 57409; direct-Frenet 3D
  surface plotting; regenerated README figures and optimizer report.
- Verification: every showcased design independently passes the Curvo profile
  with \(\lvert\iota\rvert\ge0.4\); the database source ID and URL are frozen
  in configuration and figure metadata; strong-axis surface smoothness is a
  regression test.
- Numerical checks: the derived ID-57409 case has
  \(\lvert\iota\rvert=2.964\), \(r_\mathrm{sing}=0.249\) m, and weighted
  \(B_{20}\) residual \(1.2743\times10^{-10}\) at `nphi=121`, stable through
  `nphi=481`. The finite-current showcase has a 32.99% minimum plasma
  fraction and 4.43% angular variation while passing the same screen.
- Limitation: the current-visible split uses formal radius 0.45 m and carries
  a 0.2286 T estimated field remainder. It is explicitly a field-jet
  demonstration; smaller-radius production studies require radius
  convergence and current/geometry re-optimization.
