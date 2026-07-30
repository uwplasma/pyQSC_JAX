# Vacuum coil target

Coil optimization lives downstream in ESSOS so pyQSC_JAX has no circular
dependency. In vacuum, `formal_radius=None` makes the external target exactly
the total near-axis field jet. ESSOS normalizes the field, STF gradient, and
STF Hessian residual blocks by reference field and length scales.

The executable stage-two integration is maintained in the
[ESSOS finite-beta integration PR](https://github.com/uwplasma/ESSOS/pull/46).
Its vacuum-reduction test confirms that the new objective is identical to the
total target when `I2 = 0`.

For a local paired checkout, install pyQSC_JAX first and then ESSOS:

```bash
python -m pip install -e /path/to/pyQSC_JAX
python -m pip install -e /path/to/ESSOS
python /path/to/ESSOS/examples/optimize_coils_for_near_axis_vacuum.py
```
