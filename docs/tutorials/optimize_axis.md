# Optimize an axis

This deliberately small deterministic example exercises bounded exploration,
exact \(B_{2c}\) elimination, local refinement, basin clustering, and status
semantics.

```{literalinclude} ../../examples/06_optimize_axis_B20.py
:language: python
:linenos:
```

Increase the search budget, activate modes gradually, and retain doubled- and
quadrupled-grid verification for research searches. `best_found` never means
“globally optimal.”
