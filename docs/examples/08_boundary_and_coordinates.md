# Boundary through the ESSOS adapter

Builds a second-order field with the pyQSC-style adapter
`pyqsc_jax.near_axis.near_axis` and evaluates a boundary at $r = 0.05$ m with
`get_boundary`, which returns Cartesian coordinates and the cylindrical radius
({doc}`../user_guide/interfaces`).

```{figure} ../_static/figures/example_08.png
:width: 75%
:alt: The boundary in 3-D and its \phi = 0 cross-section.

The boundary in 3-D and its $\phi = 0$ cross-section.
```

Output:

```{literalinclude} output/08.txt
:language: text
```

```{literalinclude} ../../examples/08_boundary_and_coordinates.py
:language: python
:caption: examples/08_boundary_and_coordinates.py
```
