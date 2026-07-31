# Release checklist

No release is made from an unreviewed branch.

- [x] Full line and branch coverage is at least 95%.
- [x] Ruff, all tests, all examples, docs with warnings as errors, doctests,
  and compatibility checks pass.
- [x] Wheel and sdist metadata pass `twine check`.
- [x] Wheel and sdist each install in clean environments and pass a physics
  smoke test.
- [x] Upstream pyQSC parity and high-resolution cases are rerun.
- [x] ESSOS vacuum and finite-beta integration passes in a clean environment.
- [x] Dependency and license audits have no unresolved issue.
- [x] Performance report records hardware, versions, command, and raw samples.
- [x] Changelog, migration guide, README, API docs, and limitations are current.
- [x] Publication figures are regenerated with commit/parameter metadata.
- [ ] TestPyPI trusted-publishing dry run succeeds.
- [ ] PyPI OIDC environment is configured and production publish is approved.
- [ ] `CITATION.cff` version/date and Zenodo metadata are updated.
- [x] No correctness limitation or inherited downstream failure is hidden.
