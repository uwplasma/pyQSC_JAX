# ADR: solver stack and implicit differentiation

- Status: accepted for first implementation
- Date: 2026-07-29

## Context

First order requires a converged periodic nonlinear sigma solve. Second order
requires a dense coupled linear solve. Inverse transform modes add a scalar
unknown and continuation. Global/local axis optimization may later benefit from
Levenberg–Marquardt.

The current code runs exactly five Newton iterations. Differentiating those
iterations does not represent the derivative of the converged equation and
provides no convergence evidence.

## Options considered

1. Raw JAX loops and `jax.numpy.linalg.solve`: minimal, but every implicit
   derivative and transpose rule would be local maintenance.
2. Equinox/Optimistix: capable, but adds a broad solver dependency before a
   missing capability is demonstrated.
3. SOLVAX `root_solve` and `linear_solve`: already provide JAX custom-root and
   custom-linear-solve implicit differentiation around user-supplied primal
   solvers.
4. A new SOLVAX solver API: justified only by a generic, independently tested
   need beyond pyQSC_JAX.

## Decision

Use small pyQSC_JAX primal Newton and dense linear callbacks with explicit
residual, step, convergence, iteration, and conditioning reports. Wrap
converged roots and linear solutions with SOLVAX for implicit differentiation.

Do not change SOLVAX initially. Revisit a general dense
Levenberg–Marquardt implementation only after the pyQSC_JAX optimization
requirements are concrete and the capability is demonstrably reusable.

Pseudo-arclength continuation and fold detection belong in pyQSC_JAX because
they encode near-axis branch semantics.

## Acceptance

- primal residuals satisfy configured absolute and relative tolerances;
- iteration limits produce explicit failure reports;
- JVP/VJP agree with finite differences away from singular points;
- transpose solves are tested;
- second-order conditioning is reported, especially near `iota_N = 0`;
- inverse solves detect folds instead of silently changing branches.
