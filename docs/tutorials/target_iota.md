# Target rotational transform

The inverse solve holds \(\iota\) fixed and replaces it in the sigma state by
the log-magnitude of a sign-preserving `etabar`.

```{literalinclude} ../../examples/04_target_iota.py
:language: python
:linenos:
```

The forward reconstruction checks the achieved transform. If
`branch_fold` is true, the local inverse is ill-conditioned and the
[continuation workflow](../advanced/continuation.md) should be used.
