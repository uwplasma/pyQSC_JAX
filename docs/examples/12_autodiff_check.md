# Implicit derivative check

Compares the forward-mode derivative $d\iota/d\bar\eta$ through the converged
$\sigma$ solve with a centred finite difference ({doc}`../user_guide/differentiation`).

```{figure} ../_static/figures/example_12.png
:width: 75%
:alt: The two derivative values.

The two derivative values.
```

Output:

```{literalinclude} output/12.txt
:language: text
```

```{literalinclude} ../../examples/12_autodiff_check.py
:language: python
:caption: examples/12_autodiff_check.py
```
