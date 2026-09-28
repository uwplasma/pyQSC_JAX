# Plasma and external field jets

Splits the total field jet of database configuration 52521 (finite pressure, $I_2 = 0$)
into the matched plasma part and the external vacuum target at formal radius
$a = 0.15$ m, and prints the enclosed current, the plasma field fraction, the STF traces
and the remainder indicators ({doc}`../theory/plasma_field`).

```{figure} ../_static/figures/example_10.png
:width: 75%
:alt: Total, plasma and external field in the Frenet frame. The transverse plasma and external components cancel; the total is tangential.

Total, plasma and external field in the Frenet frame. The transverse plasma and external components cancel; the total is tangential.
```

Output:

```{literalinclude} output/10.txt
:language: text
```

```{literalinclude} ../../examples/10_plasma_external_field_jet.py
:language: python
:caption: examples/10_plasma_external_field_jet.py
```
