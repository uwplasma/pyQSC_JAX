# Migration from pyQSC and the legacy adapter

## Canonical immutable API

Replace:

```python
from qsc import Qsc

stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar)
```

with:

```python
import pyqsc_jax as qsc

solution = qsc.Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar)
```

Canonical sampled arrays put `nphi` first: `B_axis` is `(nphi, 3)`,
`grad_B_axis` is `(nphi, 3, 3)`, and `grad_grad_B_axis` is
`(nphi, 3, 3, 3)`. Results are frozen. Rebuild with changed explicit inputs
instead of assigning fields.

Constructor `order` accepts `"r1"`, `"r2"`, or `"r3"`. Second-order fields
raise an attribute error on an r1 result rather than returning stale or
partially initialized values. Check `root_report`, `linear_report`, and
resolution diagnostics explicitly.

## Existing ESSOS code

Existing imports remain valid:

```python
from pyqsc_jax.near_axis import near_axis
```

The adapter preserves historical component-first arrays, mutable `x` and
`dofs`, `R0`, `Z0`, `phi`, `B_axis`, `grad_B_axis`, `iota`, boundary
conversion, and plotting methods. Its constructor and `dofs` setter now use
the same normal/binormal packing order. It delegates all physics to the
immutable core.

New code should prefer `Qsc`/`solve`. Keep the adapter only at a downstream
compatibility boundary. The package does not emit an import-time or runtime
deprecation warning while ESSOS depends on this contract.

## Behavioral differences

- even `nphi` is honored rather than silently changed;
- general `rs` and `zc` axis coefficients are supported;
- failed solves return structured nonconvergence evidence;
- sigma and r2 derivatives are implicit derivatives of converged equations;
- inverse transform solving is explicit through `solve_for`;
- nonzero global-search results never claim global optimality;
- plasma/external separation requires a positive formal radius.
