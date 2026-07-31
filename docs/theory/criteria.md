# Named design criteria

`Criteria.from_curvo_2025` implements the screening profile in Table 3 of
Curvo, Ferreira, and Jorge {cite}`curvo2025deep`. It is a named, configurable
profile rather than a package-wide definition of a “good stellarator.”

For the paper's normalization \(R_{c0}=1\,\mathrm{m}\) and
\(B_0=1\,\mathrm{T}\), the profile requires:

- positive magnetic-axis length;
- \(\lvert\iota\rvert\geq0.2\);
- maximum elongation no greater than \(10\);
- \(\min L_{\nabla B}\geq0.1\,\mathrm{m}\);
- minimum cylindrical axis radius at least \(0.3\,\mathrm{m}\);
- singular radius at least \(0.05\,\mathrm{m}\);
- \(\min L_{\nabla\nabla B}\geq0.1\,\mathrm{m}\);
- \(B_{20}\) peak-to-peak variation no greater than
  \(5\,\mathrm{T/m^2}\);
- \(\beta\geq10^{-4}\); and
- \(D_{\mathrm{Merc}}r^2>0\).

The profile uses the paper's near-axis pressure proxy

\[
\beta=-\frac{\mu_0p_2r_{\mathrm{singularity}}^2}{B_0^2}.
\]

When `major_radius` or `B0` differs from the reference normalization, length
thresholds scale with `major_radius` and the \(B_{20}\)-variation threshold
scales as \(B_0/R^2\). Dimensionless thresholds do not change. Callers can
override any threshold by name:

```python
criteria = qsc.Criteria.from_curvo_2025(
    major_radius=1.7,
    B0=2.5,
    maximum_elongation=8.0,
    minimum_beta=2.0e-4,
)
report = criteria.evaluate(solution)
```

Every `CriterionEvaluation` contains the measured value, threshold, comparison
sense, units, pass flag, and a signed raw margin. A positive margin is
favorable for both minimum and maximum criteria. `report.passed` is true only
if all ten checks pass.

The singular radius is the first coordinate-map singularity of the truncated
near-axis construction. It is useful as a screening proxy but does not prove
that a finite-radius equilibrium has nested, nonintersecting flux surfaces.
Likewise, this profile does not replace coil feasibility, fast-particle
confinement, MHD equilibrium, or free-boundary validation.
