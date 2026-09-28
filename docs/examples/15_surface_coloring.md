# Surface colouring and error fields

Plots a second-order QA surface coloured by the near-axis $|B|$ (the default of
`plot_surface_3d`) and by $\log_{10}(|B\cdot n|/|B|)$ of an external field, here a $1/R$
toroidal field with a 1 mT vertical error field. Any callable returning Cartesian $B$ at
`(n, 3)` points works, e.g. a vmapped ESSOS `BiotSavart.B` ({doc}`../user_guide/plotting`).

```{figure} ../_static/figures/example_15.png
:width: 90%
:alt: A QA surface coloured by |B| and by the relative normal field error.

Left: near-axis $|B|$. Right: relative normal field of the external field.
```

Output:

```{literalinclude} output/15.txt
:language: text
```

```{literalinclude} ../../examples/15_surface_coloring.py
:language: python
:caption: examples/15_surface_coloring.py
```
