# Custom criteria

`Criteria.from_curvo_2025` reproduces a named published profile with scalable
dimensionful thresholds. Every threshold can be overridden:

```python
import pyqsc_jax as qsc

criteria = qsc.Criteria.from_curvo_2025(
    major_radius=1.7,
    B0=2.5,
    maximum_elongation=8.0,
    minimum_beta=2.0e-4,
)
report = criteria.evaluate(solution)
for evaluation in report.evaluations:
    print(evaluation.name, evaluation.value, evaluation.margin)
```

Margins are signed so positive is favorable for both lower and upper bounds.
The profile is a screening tool, not a universal definition of feasibility.
Coil engineering, finite-radius equilibrium, and particle confinement remain
separate analyses. Definitions and normalization are in
[named design criteria](../theory/criteria.md).
