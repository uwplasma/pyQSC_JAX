# Prescribed rotational transform

Solves $\iota(\bar\eta) = 0.42$ for $\bar\eta$ by Newton iteration, with
$d\iota/d\bar\eta$ from the implicit derivative of the $\sigma$ solve
(`jax.value_and_grad`), then plots the resulting $\sigma(\varphi)$. No inverse-problem
machinery is needed ({doc}`../user_guide/differentiation`).

```{figure} ../_static/figures/example_04.png
:width: 75%
:alt: \sigma for the target transform.

$\sigma$ for the target transform.
```

Output:

```{literalinclude} output/04.txt
:language: text
```

```{literalinclude} ../../examples/04_target_iota.py
:language: python
:caption: examples/04_target_iota.py
```
