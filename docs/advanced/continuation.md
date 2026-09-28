# Continuation and folds

Target-\(\iota\) inversion is local and can be multivalued when
\(\partial\iota/\partial\bar\eta=0\). `continue_etabar_branch` instead traces
the complete sigma collocation state using a secant predictor and augmented
pseudo-arclength corrector. It preserves the sign of `etabar`.

```{literalinclude} ../../examples/11_continuation_scan.py
:language: python
:linenos:
```

`ContinuationResult` retains every corrected immutable solution, tangent,
response derivative, and fold flag. A partial branch reports
`solver_failure` or `branch_zero_crossing`; it is never labeled complete.
The [inverse/continuation derivation](../theory/inverse-solves.md) gives the
augmented equation and acceptance rules.
