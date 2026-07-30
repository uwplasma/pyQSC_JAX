# Contributing

pyQSC_JAX changes must preserve the ESSOS legacy contract and provide evidence
for every physics result.

## Developer setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,docs,plot]'
pytest
ruff check src tests
python -m sphinx -W -b html docs docs/_build/html
```

Do not pin dependency versions in package metadata or committed requirements
files. Release reports record the versions actually tested.

## Physics changes

Before merging a physics result:

1. add its primary equation source and conventions to the traceability table;
2. add a focused unit test;
3. add an independent regression or physical-identity test;
4. document shapes, units, validity assumptions, and failure modes;
5. verify JIT and automatic differentiation where the API promises them.

Reference data must be small, local, checksummed, license-compatible, and tied
to an upstream commit. Tests must never download data.

## Pull requests

Keep commits small and green. Explain the physical source, numerical method,
validation, compatibility impact, and any remaining limitation. Do not combine
unrelated formatting or generated output with physics changes.
