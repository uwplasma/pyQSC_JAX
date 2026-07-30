# Release checklist

No release is made from an unreviewed branch.

- [ ] Full line and branch coverage is at least 95%.
- [ ] Ruff, all tests, all examples, docs with warnings as errors, doctests,
  and compatibility checks pass.
- [ ] Wheel and sdist metadata pass `twine check`.
- [ ] Wheel and sdist each install in clean environments and pass a physics
  smoke test.
- [ ] Upstream pyQSC parity and high-resolution cases are rerun.
- [ ] ESSOS vacuum and finite-beta integration passes in a clean environment.
- [ ] Dependency and license audits have no unresolved issue.
- [ ] Performance report records hardware, versions, command, and raw samples.
- [ ] Changelog, migration guide, README, API docs, and limitations are current.
- [ ] Publication figures are regenerated with commit/parameter metadata.
- [ ] TestPyPI trusted-publishing dry run succeeds.
- [ ] PyPI OIDC environment is configured and production publish is approved.
- [ ] `CITATION.cff` version/date and Zenodo metadata are updated.
- [ ] No correctness limitation or inherited downstream failure is hidden.
