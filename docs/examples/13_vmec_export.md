# VMEC export

Writes a VMEC `&INDATA` file for the QA example at $r = 0.005$ m and prints the
angle-inversion and reconstruction diagnostics. The file is not run through VMEC here
({doc}`../theory/vmec`).

```{figure} ../_static/figures/example_13.png
:width: 75%
:alt: Boundary cross-sections at uniform cylindrical angles (left) and the sorted Fourier spectrum (right).

Boundary cross-sections at uniform cylindrical angles (left) and the sorted Fourier spectrum (right).
```

Output:

```{literalinclude} output/13.txt
:language: text
```

```{literalinclude} ../../examples/13_vmec_export.py
:language: python
:caption: examples/13_vmec_export.py
```
