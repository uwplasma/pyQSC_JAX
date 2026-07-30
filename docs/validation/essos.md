# ESSOS integration

ESSOS consumes pyQSC_JAX; pyQSC_JAX has no ESSOS dependency. Integration is
developed on ESSOS branch `refactor/pyqsc-external-field-jet` in draft PR
[#46](https://github.com/uwplasma/ESSOS/pull/46).

The validated integration provides:

- exact vacuum reduction to the total field jet;
- normalized smooth residual blocks for field, 5 STF-gradient, and 7
  STF-Hessian components;
- finite-current targets that subtract the plasma jet;
- actual coil-shape and current gradients checked against finite differences;
- axis and `etabar` gradients checked against finite differences;
- unchanged legacy-adapter smoke coverage;
- executable vacuum stage-two, finite-beta stage-two, and single-stage
  examples.

The focused integration suite passes 24 tests. Demonstration optimizations
reduce the finite-beta stage-two objective from 10.74 to 4.09 and the
single-stage objective from 10.75 to 4.52 at their deliberately small example
budgets. These are execution checks, not optimized-device claims.
