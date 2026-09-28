# First-order QH

The four-period quasi-helically symmetric example of {cite:t}`landreman2019numerical`.
The normal vector turns once per period, so the helicity is $N = -1$ and
$\iota_N = \iota + Nn_\mathrm{fp}$ is the transform that enters the $\sigma$ equation
({doc}`../theory/conventions`).

```{figure} ../_static/figures/example_02.png
:width: 75%
:alt: The QH magnetic axis.

The QH magnetic axis.
```

Output:

```{literalinclude} output/02.txt
:language: text
```

```{literalinclude} ../../examples/02_first_order_qh.py
:language: python
:caption: examples/02_first_order_qh.py
```
