# Coil and plasma field on a database stellarator

Downloads configuration 139379 from the [stellarator database](https://stellarator.physics.wisc.edu/)
{cite}`curvo2025deep` (network access and `requests` needed), solves it to `order="r3"`
with finite pressure and $I_2 = 0$, and draws the axis, a surface at the singular radius and
three arrow fields along the axis: the total field direction, a simple pressure-driven
plasma-field model built inline from $\beta = -\mu_0p_2r_\mathrm{sing}^2/B_0^2$, and their
difference as the coil field. The inline model is illustrative; the package's matched
plasma field is {doc}`10_plasma_external_field_jet`.

```{figure} ../_static/figures/example_00.png
:width: 75%
:alt: The axis (black), a surface at r_\mathrm{sing}, and the total (green), model plasma (blue) and coil (red) field arrows.

The axis (black), a surface at $r_\mathrm{sing}$, and the total (green), model plasma (blue) and coil (red) field arrows.
```

Output:

```{literalinclude} output/00.txt
:language: text
```

```{literalinclude} ../../examples/00_coil_field_finite_beta.py
:language: python
:caption: examples/00_coil_field_finite_beta.py
```
