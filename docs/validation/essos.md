# ESSOS integration

ESSOS consumes pyQSC_JAX; pyQSC_JAX has no ESSOS dependency. Integration is
developed on ESSOS branch `refactor/pyqsc-external-field-jet` in draft PR
[#46](https://github.com/uwplasma/ESSOS/pull/46).

The validated integration provides:

- exact vacuum reduction to the total field jet;
- normalized smooth residual blocks for field, 5 STF-gradient, and 7
  STF-Hessian components;
- finite-pressure, exactly zero-current targets that subtract the plasma jet;
- actual coil-shape and current gradients checked against finite differences;
- axis and `etabar` gradients checked against finite differences;
- unchanged legacy-adapter smoke coverage;
- executable vacuum stage-two, finite-beta stage-two, and single-stage
  examples.

The focused integration suite passes 13 tests, including both optimization
scripts in clean subprocesses. The demonstrations use the nonplanar,
finite-pressure database stellarator ID 52521 with exactly \(I_2=0\),
\(\lvert\iota\rvert>2.8\), and RMS axis torsion above
\(0.85\ {\rm m}^{-1}\).

At deliberately small example budgets, the stage-two normalized objective
decreases from \(6.7102\times10^7\) to \(4.4677\times10^3\). Its field,
gradient, and Hessian block sums all decrease independently:
\(4.4786\to0.7913\), \(2.0187\times10^4\to72.3078\), and
\(2.6833\times10^8\to1.7578\times10^4\). The single-stage objective decreases
from \(6.7102\times10^7\) to \(3.0052\times10^4\); its corresponding blocks
decrease to \(0.6382\), \(137.638\), and \(1.1966\times10^5\). The
single-stage solve keeps \(I_2=0\), reaches
\(\lvert\iota\rvert=2.843\), and lowers the weighted \(B_{20}\) residual from
0.2930 to 0.2514. These short runs establish execution and gradient coupling,
not device-quality global optima.
