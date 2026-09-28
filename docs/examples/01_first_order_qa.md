# First-order QA

The three-period quasi-axisymmetric example of {cite:t}`landreman2019numerical` at first
order: solve the $\sigma$ equation, print $\iota$, the residual, the axis length and the
maximum elongation, and plot the axis. {doc}`../theory/first_order` gives the equations.

```{figure} ../_static/figures/example_01.png
:width: 75%
:alt: The magnetic axis over the full torus.

The magnetic axis over the full torus.
```

Output:

```{literalinclude} output/01.txt
:language: text
```

```{literalinclude} ../../examples/01_first_order_qa.py
:language: python
:caption: examples/01_first_order_qa.py
```
